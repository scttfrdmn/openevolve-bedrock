#!/usr/bin/env python3
# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""
Ensemble-mix experiment: run the same problem under several model mixes and
report iterations-to-threshold and $ for each.

    python bench/matrix.py examples/circle_packing \
        --pricing pricing.yaml --iterations 100 --threshold 0.99 \
        --seeds 1 2 3 \
        --variant fast-only "haiku:1.0" \
        --variant 80-20     "haiku:0.8,sonnet:0.2" \
        --variant 50-50     "haiku:0.5,sonnet:0.5"

Model aliases used in --variant come from --alias (defaults below). Each
(variant, seed) run gets its own output dir and ledger under bench_out/.
Results accumulate in bench_out/results.csv; a row per run.

`realized_mix` is the share of ledger calls per model, in the same syntax as
the configured mix; compare the two before reading anything into a row.

Metric is `combined_score` unless --metric says otherwise. The threshold is a
metric value; "iterations to threshold" is the first checkpoint whose
best_program_info.json meets it, so set checkpoint_interval small (10).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

# Any Bedrock model id works in --variant; these are shorthands (us-west-2, 2026-10).
DEFAULT_ALIASES = {
    "haiku": "us.anthropic.claude-haiku-5-5",
    "sonnet": "us.anthropic.claude-sonnet-5-5",
    "opus": "us.anthropic.claude-opus-5-5",
    "haiku-4.5": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "nova-lite": "us.amazon.nova-2-lite-v1:0",
    "gpt-oss": "openai.gpt-oss-120b-1:0",
    "gpt-oss-20b": "openai.gpt-oss-20b-1:0",
    "deepseek": "deepseek.v3.2",
    "qwen-coder": "qwen.qwen3-coder-480b-a35b-v1:0",
    "devstral": "mistral.devstral-2-123b",
    "kimi": "moonshotai.kimi-k2.5",
    "glm": "zai.glm-5",
    "llama4": "us.meta.llama4-maverick-17b-instruct-v1:0",
}


def parse_mix(spec: str, aliases: dict) -> list:
    out = []
    for part in spec.split(","):
        name, _, w = part.strip().partition(":")
        out.append({"name": aliases.get(name, name), "weight": float(w or 1.0)})
    return out


def realized_mix(by_model: dict, configured: list, aliases: dict) -> str:
    """Share of ledger calls per model, in the MIX syntax: 'haiku:0.80,sonnet:0.20'."""
    total = sum(v.get("calls", 0) for v in by_model.values())
    if not total:
        return ""
    short = {v: k for k, v in aliases.items()}
    names = [m["name"] for m in configured]
    names += sorted(n for n in by_model if n not in names)
    return ",".join(
        f"{short.get(n, n)}:{by_model.get(n, {}).get('calls', 0) / total:.2f}" for n in names
    )


HEADER = ["variant", "mix", "realized_mix", "seed", "iterations", "best_metric",
          "iter_to_threshold", "usd_total", "usd_to_threshold", "calls", "throttled", "wall_s"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("example_dir", help="directory with initial_program.py, evaluator.py, config.yaml")
    ap.add_argument("--pricing", required=True)
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--metric", default="combined_score")
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--variant", nargs=2, action="append", metavar=("NAME", "MIX"), required=True)
    ap.add_argument("--alias", nargs=2, action="append", metavar=("ALIAS", "MODEL_ID"), default=[])
    ap.add_argument("--out", default="bench_out")
    ap.add_argument("--api-base", help="override llm.api_base (e.g. Mantle URL for the endpoint experiment)")
    args = ap.parse_args()

    aliases = dict(DEFAULT_ALIASES, **dict(args.alias))
    ex = Path(args.example_dir).resolve()
    base_cfg = yaml.safe_load((ex / "config.yaml").read_text())
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    results_csv = out_root / "results.csv"
    new_file = not results_csv.exists()
    if not new_file:
        with open(results_csv, newline="") as fh:
            existing = next(csv.reader(fh), None)
        if existing != HEADER:
            print(f"{results_csv} has different columns; use a new --out", file=sys.stderr)
            return 2

    with open(results_csv, "a", newline="") as fh:
        w = csv.writer(fh)
        if new_file:
            w.writerow(HEADER)
        for vname, mix in args.variant:
            for seed in args.seeds:
                run_dir = out_root / f"{vname}_seed{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                cfg = json.loads(json.dumps(base_cfg))  # deep copy
                cfg["random_seed"] = seed
                models = parse_mix(mix, aliases)
                cfg.setdefault("llm", {})["models"] = models
                cfg["llm"].pop("primary_model", None)
                cfg["llm"].pop("secondary_model", None)
                if args.api_base:
                    cfg["llm"]["api_base"] = args.api_base
                cfg["max_iterations"] = args.iterations
                cfg["checkpoint_interval"] = min(int(cfg.get("checkpoint_interval", 10)), 10)
                cfg_path = run_dir / "config.yaml"
                cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

                env = dict(os.environ, OE_BEDROCK_LEDGER=str(run_dir / "bedrock_ledger.jsonl"))
                t0 = time.time()
                cmd = [sys.executable, "-m", "openevolve_bedrock", "run",
                       str(ex / "initial_program.py"), str(ex / "evaluator.py"),
                       "--config", str(cfg_path), "--output", str(run_dir),
                       "--iterations", str(args.iterations)]
                print(f"\n=== {vname} seed={seed}: {' '.join(cmd)}", flush=True)
                rc = subprocess.call(cmd, env=env, cwd=str(ex))
                wall = time.time() - t0
                if rc != 0:
                    print(f"run failed rc={rc}", flush=True)

                cost = subprocess.run(
                    [sys.executable, "-m", "openevolve_bedrock", "cost", str(run_dir),
                     "--pricing", args.pricing, "--json"],
                    capture_output=True, text=True,
                )
                total = json.loads(cost.stdout) if cost.returncode == 0 else {}
                thr = subprocess.run(
                    [sys.executable, "-m", "openevolve_bedrock", "cost", str(run_dir),
                     "--pricing", args.pricing, "--json", "--metric", args.metric,
                     "--threshold", str(args.threshold)],
                    capture_output=True, text=True,
                )
                to_thr = json.loads(thr.stdout) if thr.returncode == 0 else {}
                best = None
                info = run_dir / "best" / "best_program_info.json"
                if info.is_file():
                    best = (json.loads(info.read_text()).get("metrics") or {}).get(args.metric)
                hit = to_thr.get("threshold") or {}
                row = [vname, mix, realized_mix(total.get("by_model") or {}, models, aliases),
                       seed, args.iterations, best,
                       hit.get("iteration"), total.get("usd"),
                       to_thr.get("usd") if hit else None,
                       total.get("calls"), total.get("throttled"), round(wall, 1)]
                w.writerow(row)
                fh.flush()
                print("row:", row, flush=True)
    print(f"\nresults: {results_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
