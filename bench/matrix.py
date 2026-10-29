#!/usr/bin/env python3
# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""
Ensemble-mix experiment: run the same problem under several model mixes and
score each run by best-so-far against iterations and dollars.

    python bench/matrix.py examples/circle_packing \
        --pricing pricing.yaml --iterations 100 --threshold 0.99 0.995 0.999 \
        --seeds 1 2 3 \
        --variant fast-only "haiku:1.0" \
        --variant 80-20     "haiku:0.8,sonnet:0.2" \
        --variant 50-50     "haiku:0.5,sonnet:0.5"

Model aliases used in --variant come from --alias (defaults below). Each
(variant, seed) run gets its own dir under --out with its config, ledger,
checkpoints and run_meta.json. A run whose final checkpoint exists is reused,
not rerun, so an interrupted matrix resumes where it stopped.

Thresholds only affect scoring. After the runs, results.csv and curves.csv
are rebuilt from every run dir under --out; ``--rescore`` rebuilds them without
running anything, e.g. at other thresholds.

results.csv, a row per run: best@N (best metric at the last checkpoint <= N),
then per threshold T ``iter@T`` (first checkpoint at or above T) and ``usd@T``
($ of calls completed by then), blank if not reached; then totals.
``realized_mix`` is the share of ledger calls per model, in the --variant
syntax; compare it with ``mix`` before reading anything into a row.

curves.csv, a row per checkpoint per run: best-so-far, cumulative $ and calls,
seconds since the run's first call. This is the $/result curve.

Metric is ``combined_score`` unless --metric says otherwise. Generated configs
checkpoint every 5 iterations.
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

from openevolve_bedrock import ledger as led

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
        # The weight follows the last ':', since model ids may contain one (...-v1:0).
        name, _, w = part.strip().rpartition(":")
        if not name:
            name, w = w, ""
        weight = float(w or 1.0)
        if weight <= 0:
            raise ValueError(f"{part!r}: weight must be > 0 (write ids containing ':' as ID:WEIGHT)")
        out.append({"name": aliases.get(name, name), "weight": weight})
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


CHECKPOINT_INTERVAL = 5
CURVE_HEADER = ["variant", "mix", "seed", "iteration", "best_metric", "usd_cum", "calls_cum",
                "elapsed_s"]


def _fmt(t: float) -> str:
    return f"{t:g}"


def results_header(iterations: int, thresholds: list) -> list:
    cols = ["variant", "mix", "realized_mix", "seed", "iterations", f"best@{iterations}"]
    for t in thresholds:
        cols += [f"iter@{_fmt(t)}", f"usd@{_fmt(t)}"]
    return cols + ["usd_total", "calls", "throttled", "wall_s"]


def score_run(run_dir: Path, meta: dict, pricing: dict, metric: str, iterations: int,
              thresholds: list, aliases: dict):
    """(results row, curve rows) for one run dir, from its ledger and checkpoints."""
    records = list(led.read(run_dir / "bedrock_ledger.jsonl")) \
        if (run_dir / "bedrock_ledger.jsonl").is_file() else []
    total = led.summarize(records, pricing)
    series = led.checkpoint_series(run_dir, metric)
    t0 = min((float(r.get("ts", 0)) for r in records), default=None)

    curve = []
    for p in series:
        upto = led.summarize(records, pricing, until_ts=p.reached_ts)
        curve.append([meta["variant"], meta["mix"], meta["seed"], p.iteration, p.metric_value,
                      round(upto.total_usd, 6), upto.total_calls,
                      round(p.reached_ts - t0, 1) if t0 is not None else None])

    at_n = [p for p in series if p.iteration <= iterations]
    row = [meta["variant"], meta["mix"],
           realized_mix(total.by_model, parse_mix(meta["mix"], aliases), aliases),
           meta["seed"], meta.get("iterations"), at_n[-1].metric_value if at_n else None]
    for t in thresholds:
        hit = next((c for c in curve if c[4] >= t), None)
        row += [hit[3], hit[5]] if hit else [None, None]
    row += [round(total.total_usd, 6), total.total_calls, total.throttled, meta.get("wall_s")]
    return row, curve


def run_one(ex: Path, base_cfg: dict, run_dir: Path, meta: dict, models: list,
            iterations: int, api_base) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(json.dumps(base_cfg))  # deep copy
    cfg["random_seed"] = meta["seed"]
    cfg.setdefault("llm", {})["models"] = models
    cfg["llm"].pop("primary_model", None)
    cfg["llm"].pop("secondary_model", None)
    if api_base:
        cfg["llm"]["api_base"] = api_base
    cfg["max_iterations"] = iterations
    cfg["checkpoint_interval"] = CHECKPOINT_INTERVAL
    cfg_path = run_dir / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    env = dict(os.environ, OE_BEDROCK_LEDGER=str(run_dir / "bedrock_ledger.jsonl"))
    cmd = [sys.executable, "-m", "openevolve_bedrock", "run",
           str(ex / "initial_program.py"), str(ex / "evaluator.py"),
           "--config", str(cfg_path), "--output", str(run_dir),
           "--iterations", str(iterations)]
    print(f"\n=== {meta['variant']} seed={meta['seed']}: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd, env=env, cwd=str(ex))
    meta.update(wall_s=round(time.time() - t0, 1), rc=rc)
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=1))
    if rc != 0:
        print(f"run failed rc={rc}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("example_dir", help="directory with initial_program.py, evaluator.py, config.yaml")
    ap.add_argument("--pricing", required=True)
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--metric", default="combined_score")
    ap.add_argument("--threshold", type=float, nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--variant", nargs=2, action="append", metavar=("NAME", "MIX"), default=[])
    ap.add_argument("--alias", nargs=2, action="append", metavar=("ALIAS", "MODEL_ID"), default=[])
    ap.add_argument("--out", default="bench_out")
    ap.add_argument("--api-base", help="override llm.api_base (e.g. Mantle URL for the endpoint experiment)")
    ap.add_argument("--rescore", action="store_true", help="don't run; rebuild the CSVs from --out")
    args = ap.parse_args()
    if not args.variant and not args.rescore:
        ap.error("--variant is required unless --rescore")

    aliases = dict(DEFAULT_ALIASES, **dict(args.alias))
    ex = Path(args.example_dir).resolve()
    base_cfg = yaml.safe_load((ex / "config.yaml").read_text())
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    pricing = led.load_pricing(Path(args.pricing))

    if not args.rescore:
        for vname, mix in args.variant:
            for seed in args.seeds:
                run_dir = out_root / f"{vname}_seed{seed}"
                if (run_dir / "checkpoints" / f"checkpoint_{args.iterations}").is_dir():
                    print(f"reusing {run_dir.name} (checkpoint_{args.iterations} exists)", flush=True)
                    continue
                meta = {"variant": vname, "mix": mix, "seed": seed, "iterations": args.iterations}
                run_one(ex, base_cfg, run_dir, meta, parse_mix(mix, aliases), args.iterations,
                        args.api_base)

    rows, curves = [], []
    for meta_path in sorted(out_root.glob("*/run_meta.json")):
        meta = json.loads(meta_path.read_text())
        row, curve = score_run(meta_path.parent, meta, pricing, args.metric, args.iterations,
                               args.threshold, aliases)
        rows.append(row)
        curves += curve
    for name, header, data in (("results.csv", results_header(args.iterations, args.threshold), rows),
                               ("curves.csv", CURVE_HEADER, curves)):
        with open(out_root / name, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(data)
    unpriced = sorted({m for p in out_root.glob("*/bedrock_ledger.jsonl")
                       for m in led.summarize(led.read(p), pricing).unpriced_models})
    if unpriced:
        print(f"warning: no price in {args.pricing} for {', '.join(unpriced)}", file=sys.stderr)
    print(f"\n{len(rows)} runs -> {out_root / 'results.csv'}, {out_root / 'curves.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
