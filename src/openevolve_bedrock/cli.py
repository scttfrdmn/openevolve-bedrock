# Copyright 2026 Scott Friedman
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
oe-bedrock — run OpenEvolve on Amazon Bedrock and account for what it cost.

  oe-bedrock run   PROGRAM EVALUATOR --config cfg.yaml [--iterations N]
                   [--target-score T] [--budget-usd X --pricing pricing.yaml] [...]
  oe-bedrock models [--region R] [--vendor anthropic] [--profiles]
  oe-bedrock check  --config cfg.yaml            # one tiny call per model
  oe-bedrock cost   LEDGER --pricing pricing.yaml [--output-dir DIR --metric M --threshold X]
  oe-bedrock install-hook [--remove]             # auto-import in every Python process
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from openevolve_bedrock import __version__, register
from openevolve_bedrock import ledger as led
from openevolve_bedrock import stopping
from openevolve_bedrock.ledger import DEFAULT_LEDGER_ENV, DEFAULT_LEDGER_NAME


def _p(*a, **k):
    print(*a, **k, flush=True)


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace, passthrough: List[str]) -> int:
    register()
    # Default the ledger into OpenEvolve's output directory so one run == one ledger.
    if DEFAULT_LEDGER_ENV not in os.environ:
        out = args.output or str(Path(args.initial_program).resolve().parent / "openevolve_output")
        os.environ[DEFAULT_LEDGER_ENV] = str(Path(out) / DEFAULT_LEDGER_NAME)
    _p(f"openevolve-bedrock {__version__}: ledger -> {os.environ[DEFAULT_LEDGER_ENV]}")
    if args.budget_usd is not None:
        if not args.pricing:
            _p("--budget-usd needs --pricing")
            return 2
        os.environ[stopping.BUDGET_ENV] = str(args.budget_usd)
        os.environ[stopping.PRICING_ENV] = str(Path(args.pricing).resolve())
        stopping.budget_from_env()  # fail now on a bad pricing file, not mid-run
        _p(f"budget ${args.budget_usd:g} priced from {os.environ[stopping.PRICING_ENV]}")

    from openevolve.cli import main as oe_main

    argv = ["openevolve-run", args.initial_program, args.evaluation_file]
    if args.config:
        argv += ["--config", args.config]
    if args.output:
        argv += ["--output", args.output]
    if args.iterations is not None:
        argv += ["--iterations", str(args.iterations)]
    if args.checkpoint:
        argv += ["--checkpoint", args.checkpoint]
    argv += passthrough
    sys.argv = argv
    return int(oe_main() or 0)


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------


def _bedrock_control_client(region: Optional[str], profile: Optional[str]):
    import boto3

    region = region or os.environ.get("OE_BEDROCK_REGION") or os.environ.get("AWS_REGION") \
        or os.environ.get("AWS_DEFAULT_REGION")
    if not region:
        raise SystemExit("no region: pass --region or export AWS_REGION / OE_BEDROCK_REGION")
    session = boto3.Session(profile_name=profile or os.environ.get("OE_BEDROCK_PROFILE") or None,
                            region_name=region)
    return session.client("bedrock"), region


def cmd_models(args: argparse.Namespace) -> int:
    from openevolve_bedrock.provider import _base_model_id, _vendor

    client, region = _bedrock_control_client(args.region, args.profile)
    want = (args.vendor or "").lower()

    def matches(model_id: str, provider_name: str = "") -> bool:
        return not want or want == _vendor(model_id) or want in provider_name.lower()

    resp = client.list_foundation_models(byOutputModality="TEXT")
    lifecycle: Dict[str, str] = {}
    rows = []
    for m in resp.get("modelSummaries", []):
        if "TEXT" not in (m.get("inputModalities") or []):
            continue
        mid = m["modelId"]
        life = (m.get("modelLifecycle") or {}).get("status", "")
        lifecycle[mid] = life
        types = m.get("inferenceTypesSupported") or []
        if not args.all and not {"ON_DEMAND", "INFERENCE_PROFILE"} & set(types):
            continue  # provisioned-only
        if not matches(mid, m.get("providerName", "")):
            continue
        call = "on-demand" if "ON_DEMAND" in types else "profile" if "INFERENCE_PROFILE" in types else "-"
        rows.append((mid, m.get("providerName", ""), call, life))
    rows.sort()
    _p(f"# text models in {region}  (call: on-demand = use the id; profile = use a us./global. id, see --profiles)")
    for r in rows:
        _p(f"{r[0]:<52} {r[1]:<12} {r[2]:<10} {r[3]}")

    if args.profiles:
        _p(f"\n# inference profiles in {region} for text models (use these ids for profile-only models)")
        paginator = client.get_paginator("list_inference_profiles")
        prof = []
        for page in paginator.paginate():
            for p in page.get("inferenceProfileSummaries", []):
                pid = p.get("inferenceProfileId", "")
                base = _base_model_id(pid)
                if base not in lifecycle and not args.all:
                    continue  # image, embedding, rerank ...
                if not matches(pid):
                    continue
                prof.append((pid, p.get("status", ""), lifecycle.get(base, "")))
        for r in sorted(prof):
            _p(f"{r[0]:<60} {r[1]:<8} model {r[2]}")
    return 0


# --------------------------------------------------------------------------
# check
# --------------------------------------------------------------------------


def _models_from_config(path: str):
    from openevolve.config import Config

    cfg = Config.from_yaml(path)
    seen = {}
    for m in list(cfg.llm.models) + list(cfg.llm.evaluator_models):
        if m.name and m.name not in seen:
            seen[m.name] = m
    return cfg, list(seen.values())


def cmd_check(args: argparse.Namespace) -> int:
    import asyncio

    register()
    from openevolve_bedrock.provider import BedrockLLM

    os.environ.setdefault(DEFAULT_LEDGER_ENV, "off")
    cfg, models = _models_from_config(args.config)
    if not models:
        _p("no models in config")
        return 1
    provider = cfg.llm.provider
    if provider != "bedrock":
        _p(f"warning: llm.provider is {provider!r}, expected 'bedrock'")
    ok = True
    # ~5.4K tokens: above every model's cache minimum (512-4096; Claude Haiku 4.5: 4096)
    long_system = ("You are a careful assistant. " * 900) if args.cache_probe else None
    for m in models:
        try:
            llm = BedrockLLM(m)
            # With --cache-probe the second identical call should read what the first wrote.
            for call in (1, 2) if long_system else (1,):
                t0 = time.monotonic()
                text = asyncio.run(
                    llm.generate_with_context(
                        system_message=long_system or "Reply with the single word OK.",
                        messages=[{"role": "user", "content": "Reply with the single word OK."}],
                        max_tokens=256,  # room for reasoning models
                    )
                )
                dt = (time.monotonic() - t0) * 1000
                u = llm.last_usage or {}
                tag = f" #{call}" if long_system else ""
                _p(f"OK   {m.name + tag:<60} {llm.transport.name:<9} {dt:6.0f} ms  "
                   f"in={u.get('prompt_tokens')} out={u.get('completion_tokens')} "
                   f"cache_read={u.get('cache_read_tokens')} cache_write={u.get('cache_write_tokens')}  "
                   f"-> {text.strip()[:30]!r}")
        except Exception as e:  # noqa: BLE001 - report everything
            ok = False
            _p(f"FAIL {m.name:<60} {type(e).__name__}: {e}")
    return 0 if ok else 2


# --------------------------------------------------------------------------
# cost
# --------------------------------------------------------------------------


def cmd_cost(args: argparse.Namespace) -> int:
    ledger_path = Path(args.ledger)
    if ledger_path.is_dir():
        ledger_path = ledger_path / DEFAULT_LEDGER_NAME
    if not ledger_path.is_file():
        _p(f"no ledger at {ledger_path}")
        return 1
    pricing = led.load_pricing(Path(args.pricing)) if args.pricing else {}
    records = list(led.read(ledger_path))

    until = None
    hit = None
    if args.threshold is not None:
        out_dir = Path(args.output_dir) if args.output_dir else ledger_path.parent
        hit = led.first_checkpoint_at_threshold(out_dir, args.metric, args.threshold)
        if hit is None:
            _p(f"threshold {args.metric} >= {args.threshold} not reached in {out_dir}/checkpoints")
        else:
            until = hit.reached_ts

    s = led.summarize(records, pricing, until_ts=until)
    if args.json:
        payload = {
            "ledger": str(ledger_path),
            "calls": s.total_calls,
            "usd": round(s.total_usd, 6),
            "input_tokens": s.input_tokens,
            "output_tokens": s.output_tokens,
            "cache_read_tokens": s.cache_read_tokens,
            "cache_write_tokens": s.cache_write_tokens,
            "throttled": s.throttled,
            "by_model": {k: {**v, "usd": round(v["usd"], 6)} for k, v in s.by_model.items()},
            "unpriced_models": s.unpriced_models,
            "threshold": None if hit is None else {
                "metric": args.metric, "value": hit.metric_value, "iteration": hit.iteration,
                "checkpoint": str(hit.checkpoint_dir),
            },
        }
        _p(json.dumps(payload, indent=2))
        return 0

    _p(f"ledger: {ledger_path}")
    if hit is not None:
        _p(f"first checkpoint with {args.metric} >= {args.threshold}: "
           f"iteration {hit.iteration} ({args.metric}={hit.metric_value:.6g}); "
           f"counting calls up to that point")
    _p(f"{'model':<58}{'calls':>7}{'in':>10}{'out':>9}{'cache_rd':>10}{'cache_wr':>10}{'USD':>11}")
    for model, v in sorted(s.by_model.items()):
        usd = f"{v['usd']:.4f}" if model not in s.unpriced_models else "   n/a"
        _p(f"{model:<58}{v['calls']:>7}{v['input_tokens']:>10}{v['output_tokens']:>9}"
           f"{v['cache_read_tokens']:>10}{v['cache_write_tokens']:>10}{usd:>11}")
    _p(f"{'total':<58}{s.total_calls:>7}{s.input_tokens:>10}{s.output_tokens:>9}"
       f"{s.cache_read_tokens:>10}{s.cache_write_tokens:>10}{s.total_usd:>11.4f}")
    if s.throttled:
        _p(f"throttled responses absorbed by retry: {s.throttled}")
    if s.unpriced_models:
        _p("unpriced models (add to pricing.yaml): " + ", ".join(s.unpriced_models))
    return 0


# --------------------------------------------------------------------------
# install-hook
# --------------------------------------------------------------------------


def _site_packages_dir() -> Path:
    import sysconfig

    return Path(sysconfig.get_paths()["purelib"])


def cmd_install_hook(args: argparse.Namespace) -> int:
    """
    Write a .pth file so every Python process in this environment imports
    openevolve_bedrock at startup. Only needed when running OpenEvolve via its
    own CLI (openevolve-run.py) with process-parallel workers; `oe-bedrock run`
    does not need it.
    """
    target = _site_packages_dir() / "openevolve_bedrock_autoload.pth"
    if args.remove:
        if target.exists():
            target.unlink()
            _p(f"removed {target}")
        else:
            _p(f"nothing to remove at {target}")
        return 0
    target.write_text("import openevolve_bedrock\n", encoding="utf-8")
    _p(f"wrote {target}")
    return 0


# --------------------------------------------------------------------------
# argparse
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="oe-bedrock", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"openevolve-bedrock {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run OpenEvolve with the bedrock provider registered")
    r.add_argument("initial_program")
    r.add_argument("evaluation_file")
    r.add_argument("--config", "-c")
    r.add_argument("--output", "-o")
    r.add_argument("--iterations", "-i", type=int)
    r.add_argument("--checkpoint")
    r.add_argument("--budget-usd", type=float,
                   help="stop once the ledger, priced with --pricing, reaches this many USD")
    r.add_argument("--pricing", help="pricing.yaml for --budget-usd")

    m = sub.add_parser("models", help="list text models / inference profiles in a region")
    m.add_argument("--region")
    m.add_argument("--profile", help="boto3 profile")
    m.add_argument("--vendor", help="filter by vendor, e.g. anthropic, amazon, mistral, deepseek, qwen")
    m.add_argument("--profiles", action="store_true", help="also list inference profiles")
    m.add_argument("--all", action="store_true",
                   help="include provisioned-only models and non-text profiles")

    c = sub.add_parser("check", help="make one short call per model in a config")
    c.add_argument("--config", "-c", required=True)
    c.add_argument("--cache-probe", action="store_true",
                   help="send a ~5K-token system prompt twice; the second call should read the cache")

    k = sub.add_parser("cost", help="summarize a ledger; optionally $ to a metric threshold")
    k.add_argument("ledger", help="ledger file or OpenEvolve output dir")
    k.add_argument("--pricing", help="pricing.yaml (USD per 1K tokens)")
    k.add_argument("--output-dir", help="OpenEvolve output dir with checkpoints/ (default: ledger dir)")
    k.add_argument("--metric", default="combined_score")
    k.add_argument("--threshold", type=float)
    k.add_argument("--json", action="store_true")

    h = sub.add_parser("install-hook", help="auto-import this package in every Python process")
    h.add_argument("--remove", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # `run` forwards unknown flags to openevolve-run.
    if argv and argv[0] == "run":
        args, passthrough = build_parser().parse_known_args(argv)
        return cmd_run(args, passthrough)
    args = build_parser().parse_args(argv)
    if args.cmd == "models":
        return cmd_models(args)
    if args.cmd == "check":
        return cmd_check(args)
    if args.cmd == "cost":
        return cmd_cost(args)
    if args.cmd == "install-hook":
        return cmd_install_hook(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
