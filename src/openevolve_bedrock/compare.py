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
Run named arms at equal budget and write one comparison table (issue #53).

    oe-bedrock compare arms.yaml --out compare_out/

The arms file::

    initial: examples/x/initial_program.py
    evaluator: examples/x/evaluator.py
    config: examples/x/config.yaml          # base config
    iterations: 60                          # the same for every arm: equal LLM call cap
    seeds: [0, 1, 2]                        # written into each run's config as random_seed
    pricing: pricing.yaml                   # optional, for $ columns
    verify_seeds: [0, 1, 2, 3]              # optional; needs evaluate_heldout in the evaluator
    report: [cpu_ratio]                     # optional; held-out metrics of each winner to tabulate
    env: {QISKIT_EVAL_JOBS: "10"}           # optional, for every arm
    arms:
      stock: {}
      integrity: {env: {OE_BEDROCK_FITNESS_INTEGRITY: "1"}}
      other_models: {config: examples/x/config_b.yaml}
      best_of_n: {kind: bestofn}          # `iterations` independent samples, no population

An arm is an OpenEvolve run unless ``kind: bestofn``, which makes the same number of LLM calls
as independent samples from the initial program (``oe-bedrock bestofn``). Otherwise arms differ
only by a config file and environment variables. OpenEvolve drops unknown YAML
keys, so that's the only safe way to vary them. Every run logs each evaluation's outcome
(``OE_BEDROCK_OUTCOMES=1``, which changes nothing about the run). That gives, per run:

- **best_train:** the run's best score;
- **heldout:** the winner's held-out mean (``verify``) and whether it was rejected;
- **verified:** the winner passed held-out correctness and beat the initial program's held-out mean;
- **report** metrics: the winner's mean over held-out seeds of each metric named in ``report``
  (default ``[cpu_ratio]``); a column appears only if some winner's evaluator returned it;
- **calls**, **llm_usd:** from the ledger;
- **evals:** evaluation attempts;
- **dup_rate:** the share of evaluations whose whitespace-normalised code was already evaluated in the run;
- **repeat_fail_rate:** the share of candidate failures whose error (digits removed) an earlier
  failure in the run already had.

Per arm, the table gives the mean ± sd over seeds, the verified rate, and $ per verified
improvement.
"""

from __future__ import annotations

import json
import os
import re
import statistics as S
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from openevolve_bedrock import integrity
from openevolve_bedrock import ledger as _ledger


DEFAULT_REPORT = ["cpu_ratio"]


@dataclass
class RunResult:
    arm: str
    seed: int
    best_train: float = 0.0
    heldout: Optional[float] = None
    rejected: bool = False
    verified: bool = False
    calls: int = 0
    llm_usd: Optional[float] = None
    evals: int = 0
    dup_rate: Optional[float] = None
    repeat_fail_rate: Optional[float] = None
    error: Optional[str] = None
    extra: Dict[str, float] = field(default_factory=dict)


def _norm_error(e: str) -> str:
    return re.sub(r"[0-9a-f]{8,}|\d+(\.\d+)?", "#", e or "").strip()[:120]


def run_stats(records: List[dict]) -> Dict[str, Optional[float]]:
    """Duplicate and repeat-failure rates from a run's outcome lines."""
    first = [r for r in records if r.get("kind") == "outcome" and r.get("attempt", 0) == 0]
    seen, dups = set(), 0
    fails, seen_err, repeats = 0, set(), 0
    for r in first:
        h = r.get("code_hash")
        if h in seen:
            dups += 1
        seen.add(h)
        if r["class"] == integrity.CANDIDATE_FAILED:
            fails += 1
            e = _norm_error(r.get("error", ""))
            if e in seen_err:
                repeats += 1
            seen_err.add(e)
    return {"evals": len(first), "dup_rate": dups / len(first) if first else None,
            "repeat_fail_rate": repeats / fails if fails else None}


def _config_for(base: Path, seed: int, out: Path, iterations: int) -> Path:
    cfg = yaml.safe_load(base.read_text()) or {}
    cfg["random_seed"] = seed
    # OpenEvolve checkpoints only every checkpoint_interval iterations, and verify reads the last
    # checkpoint: if the interval doesn't divide the run, the last iterations would go unverified.
    if iterations % int(cfg.get("checkpoint_interval") or 100):
        cfg["checkpoint_interval"] = iterations
    f = out / "config.yaml"
    f.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return f


def _one(spec: dict, root: Path, arm: str, armspec: dict, seed: int, initial_heldout: Optional[float]) -> RunResult:
    out = root / arm / f"s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    res = RunResult(arm, seed)
    led = out / "ledger.jsonl"
    env = {**os.environ, **{k: str(v) for k, v in (spec.get("env") or {}).items()},
           **{k: str(v) for k, v in (armspec.get("env") or {}).items()},
           "OE_BEDROCK_LEDGER": str(led), integrity.OBSERVE_ENV: "1"}
    cfg = _config_for(Path(armspec.get("config") or spec["config"]), seed, out, int(spec["iterations"]))
    if not (out / "run" / "best" / "best_program_info.json").exists():
        kind = armspec.get("kind", "openevolve")
        if kind not in ("openevolve", "bestofn"):
            res.error = f"unknown arm kind {kind!r}"
            return res
        cmd = (["run", spec["initial"], spec["evaluator"], "--config", str(cfg), "--output", str(out / "run"),
                "--iterations", str(spec["iterations"])] if kind == "openevolve" else
               ["bestofn", spec["initial"], spec["evaluator"], "--config", str(cfg), "--output", str(out / "run"),
                "--samples", str(spec["iterations"])])
        p = subprocess.run([sys.executable, "-m", "openevolve_bedrock", *cmd], capture_output=True, text=True, env=env)
        (out / "run.log").write_text(p.stdout + p.stderr)
        if p.returncode != 0:
            res.error = (p.stderr or p.stdout)[-500:]
            return res
    info = json.loads((out / "run" / "best" / "best_program_info.json").read_text())
    res.best_train = float(info["metrics"].get("combined_score") or 0.0)
    recs = list(_ledger.read(led)) if led.exists() else []
    st = run_stats(recs)
    res.evals, res.dup_rate, res.repeat_fail_rate = st["evals"], st["dup_rate"], st["repeat_fail_rate"]
    res.calls = sum(1 for r in recs if not r.get("kind"))
    if spec.get("pricing"):
        res.llm_usd = _ledger.summarize(recs, _ledger.load_pricing(Path(spec["pricing"]))).total_usd
    if spec.get("verify_seeds"):
        from openevolve_bedrock import verify
        old = os.environ.copy()
        os.environ.update({k: v for k, v in env.items() if k != "OE_BEDROCK_LEDGER"})
        try:
            v = verify.run(str(out / "run"), spec["evaluator"], top=1, seeds=spec["verify_seeds"])[0]
        finally:
            os.environ.clear()
            os.environ.update(old)
        res.heldout, res.rejected = v.mean, v.rejected
        for key in spec.get("report", DEFAULT_REPORT):
            res.extra[key] = v.metric_mean(key)
        res.verified = (not v.rejected) and initial_heldout is not None and v.mean is not None \
            and v.mean > initial_heldout
    return res


def _initial_heldout(spec: dict) -> Optional[float]:
    if not spec.get("verify_seeds"):
        return None
    from openevolve_bedrock.selftest import _module
    old = os.environ.copy()
    os.environ.update({k: str(v) for k, v in (spec.get("env") or {}).items()})
    try:
        mod = _module(Path(spec["evaluator"]).resolve())
        scores = [float(mod.evaluate_heldout(spec["initial"], s).get("combined_score") or 0) for s in spec["verify_seeds"]]
    finally:
        os.environ.clear()
        os.environ.update(old)
    return S.mean(scores)


def run(arms_file: str, out_dir: str) -> dict:
    spec = yaml.safe_load(Path(arms_file).read_text())
    base = Path(arms_file).resolve().parent
    for k in ("initial", "evaluator", "config", "pricing"):
        if spec.get(k):
            spec[k] = str((base / spec[k]).resolve())
    for a in spec["arms"].values():
        if a and a.get("config"):
            a["config"] = str((base / a["config"]).resolve())
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    ih = _initial_heldout(spec)
    # seed-major, so a change in machine load over a long comparison falls on every arm alike
    results = [_one(spec, root, arm, armspec or {}, seed, ih)
               for seed in spec["seeds"] for arm, armspec in spec["arms"].items()]
    summary = {"initial_heldout": ih, "arms": {}}
    for arm in spec["arms"]:
        rs = [r for r in results if r.arm == arm and not r.error]

        def ms(xs):
            xs = [x for x in xs if x is not None]
            return (S.mean(xs), S.stdev(xs) if len(xs) > 1 else 0.0) if xs else (None, None)

        usd = [r.llm_usd for r in rs if r.llm_usd is not None]
        verified = sum(r.verified for r in rs)
        summary["arms"][arm] = {
            "runs": len(rs), "failed_runs": sum(1 for r in results if r.arm == arm and r.error),
            "best_train": ms([r.best_train for r in rs]), "heldout": ms([r.heldout for r in rs]),
            "verified_rate": verified / len(rs) if rs else None, "rejected": sum(r.rejected for r in rs),
            "calls": ms([r.calls for r in rs]), "llm_usd": ms(usd) if usd else (None, None),
            "usd_per_verified": (sum(usd) / verified) if usd and verified else None,
            "evals": ms([r.evals for r in rs]), "dup_rate": ms([r.dup_rate for r in rs]),
            "repeat_fail_rate": ms([r.repeat_fail_rate for r in rs]),
            "report": {k: ms([r.extra.get(k) for r in rs]) for k in spec.get("report", DEFAULT_REPORT)},
        }
    summary["runs"] = [r.__dict__ for r in results]
    (root / "results.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


def table(summary: dict) -> str:
    def f(v, d=4):
        m, s = v
        return "—" if m is None else f"{m:.{d}f} ± {s:.{d}f}"

    lines = [f"initial program heldout: {summary['initial_heldout']}" if summary.get("initial_heldout") is not None
             else "no heldout (evaluator without evaluate_heldout, or no verify_seeds)",
             f"{'arm':18} {'runs':>4} {'best train':>17} {'heldout':>17} {'verified':>8} {'calls':>11} "
             f"{'LLM $':>15} {'$/verified':>10} {'dup rate':>13} {'repeat fail':>13}"]
    keys = [k for k in dict.fromkeys(k for a in summary["arms"].values() for k in a.get("report", {}))
            if any(a["report"][k][0] is not None for a in summary["arms"].values() if k in a.get("report", {}))]
    lines[-1] += "".join(f" {k[:15]:>15}" for k in keys)
    for arm, a in summary["arms"].items():
        vr = "—" if a["verified_rate"] is None else f"{a['verified_rate']:.0%}"
        upv = "—" if a["usd_per_verified"] is None else f"{a['usd_per_verified']:.3f}"
        lines.append(f"{arm:18} {a['runs']:>4} {f(a['best_train']):>17} {f(a['heldout']):>17} {vr:>8} "
                     f"{f(a['calls'], 0):>11} {f(a['llm_usd'], 3):>15} {upv:>10} {f(a['dup_rate'], 2):>13} "
                     f"{f(a['repeat_fail_rate'], 2):>13}"
                     + "".join(f" {f(a['report'].get(k, (None, None)), 2):>15}" for k in keys))
    return "\n".join(lines)
