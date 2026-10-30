# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""
Per-call usage ledger (JSONL) and cost computation.

One line per Bedrock call, appended with O_APPEND so OpenEvolve's spawned
worker processes can share one file without coordination (lines are far
below PIPE_BUF).

Record fields:
    ts            epoch seconds (float) at call completion
    model         model id or inference profile id as sent to Bedrock
    transport     "converse" | "openai"
    input_tokens, output_tokens, cache_read_tokens, cache_write_tokens
    latency_ms    wall time of the successful attempt
    attempts      total attempts including the successful one
    throttled     number of throttling responses before success
    stop_reason   as reported by the API, if any
    role          "evolve" | "evaluate" | "unknown" (which ensemble this model
                  serves, when known)

Evaluations run remotely (``pool.py``) add lines with ``kind: "eval"``:
    ts, kind, backend ("spawn-pool"), run_id, task_id, state, exit_code,
    eval_s (wall seconds from submit to result), cost_usd (spawn's
    ``cost_estimate`` for the task, if it had one), retry_class.
LLM-call lines have no ``kind``.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import yaml

DEFAULT_LEDGER_ENV = "OE_BEDROCK_LEDGER"
DEFAULT_LEDGER_NAME = "bedrock_ledger.jsonl"


@dataclass
class Usage:
    model: str
    transport: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    latency_ms: float = 0.0
    attempts: int = 1
    throttled: int = 0
    stop_reason: Optional[str] = None
    role: str = "unknown"
    ts: float = field(default_factory=time.time)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def as_openevolve_usage(self) -> Dict[str, object]:
        """Shape OpenEvolve expects on ``LLMInterface.last_usage``."""
        return {
            "prompt_tokens": self.input_tokens,
            "completion_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "model": self.model,
            # extra, harmless to OpenEvolve, useful to us
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
        }


def ledger_path() -> Optional[Path]:
    """Resolve the ledger path from the environment; None disables logging."""
    value = os.environ.get(DEFAULT_LEDGER_ENV)
    if value is None:
        return Path.cwd() / DEFAULT_LEDGER_NAME
    if value.strip() in ("", "0", "off", "none"):
        return None
    return Path(value).expanduser()


def append(record, path: Optional[Path] = None) -> None:
    """Append one record: a ``Usage``, or a dict for other kinds (``kind`` set)."""
    path = ledger_path() if path is None else path
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    data = record if isinstance(record, dict) else asdict(record)
    line = json.dumps(data, separators=(",", ":")) + "\n"
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line.encode("utf-8"))
    finally:
        os.close(fd)


def read(path: Path) -> Iterator[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# --------------------------------------------------------------------------
# Pricing and cost
# --------------------------------------------------------------------------

PRICE_KEYS = ("input", "output", "cache_read", "cache_write")


@dataclass
class Price:
    """USD per 1,000 tokens."""

    input: float
    output: float
    cache_read: float = 0.0
    cache_write: float = 0.0

    def cost(self, rec: Dict[str, object]) -> float:
        return (
            float(rec.get("input_tokens", 0)) * self.input
            + float(rec.get("output_tokens", 0)) * self.output
            + float(rec.get("cache_read_tokens", 0)) * self.cache_read
            + float(rec.get("cache_write_tokens", 0)) * self.cache_write
        ) / 1000.0


def load_pricing(path: Path) -> Dict[str, Price]:
    with open(path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    models = doc.get("models", {}) or {}
    out: Dict[str, Price] = {}
    for model_id, entry in models.items():
        if entry is None:
            continue
        if any(entry.get(k) is None for k in ("input", "output")):
            continue  # unfilled template row
        out[str(model_id)] = Price(
            input=float(entry["input"]),
            output=float(entry["output"]),
            cache_read=float(entry.get("cache_read") or 0.0),
            cache_write=float(entry.get("cache_write") or 0.0),
        )
    return out


def _match_price(model: str, pricing: Dict[str, Price]) -> Optional[Price]:
    """Exact id first; then strip a cross-region prefix (us./eu./apac./global.)."""
    if model in pricing:
        return pricing[model]
    for prefix in ("us.", "eu.", "apac.", "global.", "jp.", "au.", "ca."):
        if model.startswith(prefix) and model[len(prefix):] in pricing:
            return pricing[model[len(prefix):]]
    return None


@dataclass
class CostSummary:
    by_model: Dict[str, Dict[str, float]]
    total_usd: float
    total_calls: int
    unpriced_models: List[str]
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    throttled: int
    evals: int = 0
    eval_usd: float = 0.0
    eval_s: float = 0.0
    evals_unpriced: int = 0


def summarize(
    records: Iterable[Dict[str, object]],
    pricing: Dict[str, Price],
    until_ts: Optional[float] = None,
) -> CostSummary:
    by_model: Dict[str, Dict[str, float]] = {}
    unpriced: set = set()
    total = 0.0
    calls = 0
    in_t = out_t = cr_t = cw_t = thr = 0
    evals = evals_unpriced = 0
    eval_usd = eval_s = 0.0
    for rec in records:
        if until_ts is not None and float(rec.get("ts", 0)) > until_ts:
            continue
        if rec.get("kind") == "eval":
            evals += 1
            eval_s += float(rec.get("eval_s") or 0.0)
            if rec.get("cost_usd") is None:
                evals_unpriced += 1
            else:
                eval_usd += float(rec["cost_usd"])
            continue
        if rec.get("kind"):
            continue  # unknown record kinds are not LLM calls
        model = str(rec.get("model"))
        slot = by_model.setdefault(
            model,
            {"calls": 0, "input_tokens": 0, "output_tokens": 0,
             "cache_read_tokens": 0, "cache_write_tokens": 0, "usd": 0.0},
        )
        slot["calls"] += 1
        for k in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens"):
            slot[k] += int(rec.get(k, 0) or 0)
        price = _match_price(model, pricing)
        if price is None:
            unpriced.add(model)
        else:
            c = price.cost(rec)
            slot["usd"] += c
            total += c
        calls += 1
        in_t += int(rec.get("input_tokens", 0) or 0)
        out_t += int(rec.get("output_tokens", 0) or 0)
        cr_t += int(rec.get("cache_read_tokens", 0) or 0)
        cw_t += int(rec.get("cache_write_tokens", 0) or 0)
        thr += int(rec.get("throttled", 0) or 0)
    return CostSummary(
        by_model=by_model,
        total_usd=total,
        total_calls=calls,
        unpriced_models=sorted(unpriced),
        input_tokens=in_t,
        output_tokens=out_t,
        cache_read_tokens=cr_t,
        cache_write_tokens=cw_t,
        throttled=thr,
        evals=evals,
        eval_usd=eval_usd,
        eval_s=eval_s,
        evals_unpriced=evals_unpriced,
    )


# --------------------------------------------------------------------------
# $/result: first checkpoint reaching a metric threshold
# --------------------------------------------------------------------------


@dataclass
class ThresholdHit:
    checkpoint_dir: Path
    iteration: int
    metric_value: float
    reached_ts: float


def checkpoint_series(output_dir: Path, metric: str) -> List[ThresholdHit]:
    """
    Every ``<output_dir>/checkpoints/checkpoint_N/best_program_info.json`` that
    has ``metrics[metric]``, in iteration order: the best-so-far curve. The info
    file's mtime is used as the time the checkpoint was reached.
    """
    ckpt_root = output_dir / "checkpoints"
    if not ckpt_root.is_dir():
        return []
    dirs = []
    for d in ckpt_root.iterdir():
        if d.is_dir() and d.name.startswith("checkpoint_"):
            try:
                dirs.append((int(d.name.split("_", 1)[1]), d))
            except ValueError:
                continue
    out = []
    for iteration, d in sorted(dirs):
        info = d / "best_program_info.json"
        if not info.is_file():
            continue
        try:
            data = json.loads(info.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        value = (data.get("metrics") or {}).get(metric)
        if value is None:
            continue
        try:
            fv = float(value)
        except (TypeError, ValueError):
            continue
        out.append(ThresholdHit(d, iteration, fv, info.stat().st_mtime))
    return out


def first_checkpoint_at_threshold(
    output_dir: Path, metric: str, threshold: float
) -> Optional[ThresholdHit]:
    """The first checkpoint in ``checkpoint_series`` whose metric is >= threshold."""
    for point in checkpoint_series(output_dir, metric):
        if point.metric_value >= threshold:
            return point
    return None
