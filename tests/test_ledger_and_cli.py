# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
import json
import os
import time
from pathlib import Path

import pytest

from openevolve_bedrock import ledger as led
from openevolve_bedrock.cli import main as cli_main


def _write(path: Path, recs):
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))


def _rec(model, inp, out, cr=0, cw=0, ts=None, throttled=0):
    return {"ts": ts or time.time(), "model": model, "transport": "converse",
            "input_tokens": inp, "output_tokens": out, "cache_read_tokens": cr,
            "cache_write_tokens": cw, "latency_ms": 1.0, "attempts": 1,
            "throttled": throttled, "stop_reason": "end_turn", "role": "unknown"}


PRICING = """
models:
  anthropic.claude-haiku-4-5-20251001-v1:0: {input: 0.001, output: 0.005, cache_read: 0.0001, cache_write: 0.00125}
  amazon.nova-lite-v1:0: {input: null, output: null}
"""


def test_pricing_prefix_match_and_cost(tmp_path):
    p = tmp_path / "pricing.yaml"
    p.write_text(PRICING)
    pricing = led.load_pricing(p)
    assert "amazon.nova-lite-v1:0" not in pricing  # null rows ignored
    recs = [
        _rec("us.anthropic.claude-haiku-4-5-20251001-v1:0", 1000, 200, cr=500, cw=100),
        _rec("us.amazon.nova-lite-v1:0", 10, 10),
    ]
    s = led.summarize(recs, pricing)
    # 1000*0.001 + 200*0.005 + 500*0.0001 + 100*0.00125 = 1 + 1 + 0.05 + 0.125 = 2.175 per 1K -> /1000
    assert s.total_usd == pytest.approx(2.175 / 1000)
    assert s.unpriced_models == ["us.amazon.nova-lite-v1:0"]
    assert s.total_calls == 2 and s.input_tokens == 1010


def test_until_ts_filter():
    recs = [_rec("m", 1, 1, ts=100.0), _rec("m", 1, 1, ts=200.0)]
    s = led.summarize(recs, {}, until_ts=150.0)
    assert s.total_calls == 1


def test_first_checkpoint_at_threshold(tmp_path):
    ck = tmp_path / "checkpoints"
    for i, score in [(10, 0.5), (20, 0.9), (30, 0.99), (40, 1.0)]:
        d = ck / f"checkpoint_{i}"
        d.mkdir(parents=True)
        (d / "best_program_info.json").write_text(json.dumps({"metrics": {"combined_score": score}}))
        os.utime(d / "best_program_info.json", (1000 + i, 1000 + i))
    hit = led.first_checkpoint_at_threshold(tmp_path, "combined_score", 0.95)
    assert hit.iteration == 30 and hit.metric_value == 0.99
    assert hit.reached_ts == pytest.approx(1030)
    assert led.first_checkpoint_at_threshold(tmp_path, "combined_score", 2.0) is None


def test_cost_cli_json_with_threshold(tmp_path, capsys):
    out = tmp_path / "openevolve_output"
    (out / "checkpoints" / "checkpoint_10").mkdir(parents=True)
    info = out / "checkpoints" / "checkpoint_10" / "best_program_info.json"
    info.write_text(json.dumps({"metrics": {"combined_score": 0.97}}))
    os.utime(info, (5000, 5000))
    _write(out / "bedrock_ledger.jsonl", [
        _rec("anthropic.claude-haiku-4-5-20251001-v1:0", 1000, 100, ts=4000.0),
        _rec("anthropic.claude-haiku-4-5-20251001-v1:0", 1000, 100, ts=6000.0, throttled=2),
    ])
    pricing = tmp_path / "pricing.yaml"
    pricing.write_text(PRICING)

    rc = cli_main(["cost", str(out), "--pricing", str(pricing), "--json"])
    assert rc == 0
    full = json.loads(capsys.readouterr().out)
    assert full["calls"] == 2 and full["throttled"] == 2
    assert full["usd"] == pytest.approx(2 * (1000 * 0.001 + 100 * 0.005) / 1000)

    rc = cli_main(["cost", str(out), "--pricing", str(pricing), "--json", "--threshold", "0.95"])
    assert rc == 0
    part = json.loads(capsys.readouterr().out)
    assert part["calls"] == 1                      # only the call before the checkpoint
    assert part["threshold"]["iteration"] == 10


def test_ledger_append_is_line_oriented(tmp_path, monkeypatch):
    p = tmp_path / "l.jsonl"
    monkeypatch.setenv("OE_BEDROCK_LEDGER", str(p))
    led.append(led.Usage(model="m", transport="converse", input_tokens=1, output_tokens=2))
    led.append(led.Usage(model="m", transport="converse", input_tokens=3, output_tokens=4))
    recs = list(led.read(p))
    assert [r["input_tokens"] for r in recs] == [1, 3]


def test_ledger_off(tmp_path, monkeypatch):
    monkeypatch.setenv("OE_BEDROCK_LEDGER", "off")
    assert led.ledger_path() is None
    led.append(led.Usage(model="m", transport="converse"))  # no-op, no error


def test_bench_realized_mix_from_cost_json(tmp_path, capsys):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "bench_matrix", Path(__file__).resolve().parent.parent / "bench" / "matrix.py")
    matrix = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matrix)

    haiku, sonnet = matrix.DEFAULT_ALIASES["haiku"], matrix.DEFAULT_ALIASES["sonnet"]
    _write(tmp_path / "bedrock_ledger.jsonl",
           [_rec(haiku, 10, 1)] * 3 + [_rec(sonnet, 10, 1)] + [_rec("other.model-v1:0", 10, 1)])
    pricing = tmp_path / "pricing.yaml"
    pricing.write_text(PRICING)
    assert cli_main(["cost", str(tmp_path), "--pricing", str(pricing), "--json"]) == 0
    by_model = json.loads(capsys.readouterr().out)["by_model"]

    configured = matrix.parse_mix("haiku:0.8,sonnet:0.2", matrix.DEFAULT_ALIASES)
    assert matrix.realized_mix(by_model, configured, matrix.DEFAULT_ALIASES) == \
        "haiku:0.60,sonnet:0.20,other.model-v1:0:0.20"
    assert matrix.realized_mix({}, configured, matrix.DEFAULT_ALIASES) == ""


def _load_matrix():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "bench_matrix", Path(__file__).resolve().parent.parent / "bench" / "matrix.py")
    matrix = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matrix)
    return matrix


def _fake_run(run_dir: Path, model: str, points, t0=1_000_000.0):
    """points: (iteration, best metric); one call just before each checkpoint."""
    recs = []
    for k, (iteration, value) in enumerate(points):
        ts = t0 + 10 * k
        recs.append(_rec(model, 1000, 100, ts=ts))
        d = run_dir / "checkpoints" / f"checkpoint_{iteration}"
        d.mkdir(parents=True)
        info = d / "best_program_info.json"
        info.write_text(json.dumps({"metrics": {"combined_score": value}}))
        os.utime(info, (ts + 1, ts + 1))
    _write(run_dir / "bedrock_ledger.jsonl", recs)


def test_bench_scores_thresholds_and_curve(tmp_path):
    matrix = _load_matrix()
    model = "anthropic.claude-haiku-4-5-20251001-v1:0"
    _fake_run(tmp_path / "r", model, [(5, 0.90), (10, 0.991), (15, 0.993), (20, 0.996)])
    pricing = tmp_path / "pricing.yaml"
    pricing.write_text(PRICING)
    meta = {"variant": "v", "mix": f"{model}:1.0", "seed": 1, "iterations": 15, "wall_s": 42.0}
    row, curve = matrix.score_run(tmp_path / "r", meta, led.load_pricing(pricing),
                                  "combined_score", 15, [0.99, 0.995, 0.999], {})
    header = matrix.results_header(15, [0.99, 0.995, 0.999])
    r = dict(zip(header, row))
    per_call = 1000 * 0.001 / 1000 + 100 * 0.005 / 1000  # pricing is per 1K
    assert r["best@15"] == 0.993                          # not the iteration-20 value
    assert (r["iter@0.99"], r["usd@0.99"]) == (10, round(2 * per_call, 6))
    assert (r["iter@0.995"], r["usd@0.995"]) == (20, round(4 * per_call, 6))
    assert (r["iter@0.999"], r["usd@0.999"]) == (None, None)
    assert (r["calls"], r["wall_s"], r["realized_mix"]) == (4, 42.0, f"{model}:1.00")
    assert [c[3] for c in curve] == [5, 10, 15, 20]
    assert [c[6] for c in curve] == [1, 2, 3, 4]
    assert [c[7] for c in curve] == [1.0, 11.0, 21.0, 31.0]


def test_bench_rescore_rebuilds_csvs_from_run_dirs(tmp_path, monkeypatch):
    import csv as _csv
    import sys

    matrix = _load_matrix()
    model = "anthropic.claude-haiku-4-5-20251001-v1:0"
    out = tmp_path / "out"
    for seed in (1, 2):
        run = out / f"v_seed{seed}"
        _fake_run(run, model, [(5, 0.98), (10, 0.99 + seed / 1000)])
        (run / "run_meta.json").write_text(json.dumps(
            {"variant": "v", "mix": f"{model}:1.0", "seed": seed, "iterations": 10, "wall_s": 1.0}))
    ex = tmp_path / "ex"
    ex.mkdir()
    (ex / "config.yaml").write_text("llm: {}\n")
    pricing = tmp_path / "pricing.yaml"
    pricing.write_text(PRICING)
    monkeypatch.setattr(sys, "argv", ["matrix.py", str(ex), "--pricing", str(pricing),
                                      "--iterations", "10", "--threshold", "0.992",
                                      "--out", str(out), "--rescore"])
    monkeypatch.setattr(matrix.subprocess, "call", lambda *a, **k: pytest.fail("ran OpenEvolve"))
    assert matrix.main() == 0
    rows = list(_csv.DictReader(open(out / "results.csv")))
    assert [(r["seed"], r["iter@0.992"]) for r in rows] == [("1", ""), ("2", "10")]
    assert len(list(_csv.DictReader(open(out / "curves.csv")))) == 4


def test_bench_reuses_finished_runs(tmp_path, monkeypatch):
    import sys

    matrix = _load_matrix()
    out = tmp_path / "out"
    (out / "v_seed1" / "checkpoints" / "checkpoint_10").mkdir(parents=True)
    ex = tmp_path / "ex"
    ex.mkdir()
    (ex / "config.yaml").write_text("llm: {}\n")
    pricing = tmp_path / "pricing.yaml"
    pricing.write_text(PRICING)
    ran = []
    monkeypatch.setattr(matrix, "run_one", lambda ex, cfg, run_dir, *a: ran.append(run_dir.name))
    monkeypatch.setattr(sys, "argv", ["matrix.py", str(ex), "--pricing", str(pricing),
                                      "--iterations", "10", "--threshold", "0.99", "--out", str(out),
                                      "--seeds", "1", "2", "--variant", "v", "haiku:1.0"])
    assert matrix.main() == 0
    assert ran == ["v_seed2"]


def test_bench_parse_mix_accepts_versioned_ids():
    matrix = _load_matrix()
    a = matrix.DEFAULT_ALIASES
    assert matrix.parse_mix("haiku:0.8,openai.gpt-oss-120b-1:0:0.2", a) == [
        {"name": a["haiku"], "weight": 0.8}, {"name": "openai.gpt-oss-120b-1:0", "weight": 0.2}]
    assert matrix.parse_mix("gpt-oss", a) == [{"name": a["gpt-oss"], "weight": 1.0}]
    with pytest.raises(ValueError, match="ID:WEIGHT"):
        matrix.parse_mix("openai.gpt-oss-120b-1:0", a)


def test_bench_moves_interrupted_run_aside_and_reuses_stopped_ones(tmp_path, monkeypatch):
    import sys

    matrix = _load_matrix()
    out = tmp_path / "out"
    stopped = out / "v_seed1"          # stopped early at the target: no checkpoint_10
    (stopped / "checkpoints" / "checkpoint_5").mkdir(parents=True)
    (stopped / "run_meta.json").write_text(json.dumps({"variant": "v", "mix": "haiku:1.0", "seed": 1,
                                                       "iterations": 10, "rc": 0}))
    broken = out / "v_seed2"           # killed mid-run: partial ledger, no run_meta.json
    (broken / "checkpoints" / "checkpoint_5").mkdir(parents=True)
    _write(broken / "bedrock_ledger.jsonl", [_rec("m", 1, 1)])
    ex = tmp_path / "ex"
    ex.mkdir()
    (ex / "config.yaml").write_text("llm: {}\n")
    pricing = tmp_path / "pricing.yaml"
    pricing.write_text(PRICING)
    ran = []
    monkeypatch.setattr(matrix, "run_one", lambda ex, cfg, run_dir, meta, *a: ran.append((run_dir.name, a[-1])))
    monkeypatch.setattr(sys, "argv", ["matrix.py", str(ex), "--pricing", str(pricing),
                                      "--iterations", "10", "--threshold", "0.99", "--out", str(out),
                                      "--seeds", "1", "2", "--variant", "v", "haiku:1.0",
                                      "--target-score", "1.0", "--budget-usd", "2"])
    assert matrix.main() == 0
    assert ran == [("v_seed2", ["--target-score", "1.0", "--budget-usd", "2.0",
                                "--pricing", str(pricing.resolve())])]
    assert not broken.exists()
    assert len(list(out.glob("v_seed2.interrupted-*"))) == 1


def _pricing(tmp_path):
    p = tmp_path / "pricing.yaml"
    p.write_text(PRICING)
    return p


def test_eval_records_are_summed_apart_from_llm_calls(tmp_path, capsys):
    model = "anthropic.claude-haiku-4-5-20251001-v1:0"
    recs = [_rec(model, 1000, 100, ts=10.0),
            {"ts": 11.0, "kind": "eval", "backend": "spawn-pool", "task_id": "a",
             "state": "completed", "eval_s": 120.0, "cost_usd": 0.05},
            {"ts": 12.0, "kind": "eval", "backend": "spawn-pool", "task_id": "b",
             "state": "failed", "eval_s": 30.0, "cost_usd": None},
            {"ts": 13.0, "kind": "something-new"}]
    s = led.summarize(recs, led.load_pricing(_pricing(tmp_path)))
    assert s.total_calls == 1 and list(s.by_model) == [model] and not s.unpriced_models
    assert (s.evals, s.eval_usd, s.eval_s, s.evals_unpriced) == (2, 0.05, 150.0, 1)
    assert led.summarize(recs, {}, until_ts=11.0).evals == 1

    _write(tmp_path / "bedrock_ledger.jsonl", recs)
    assert cli_main(["cost", str(tmp_path), "--pricing", str(_pricing(tmp_path)), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["evals"] == {"count": 2, "usd": 0.05, "seconds": 150.0, "unpriced": 1}
    assert out["usd_with_evals"] == pytest.approx(out["usd"] + 0.05)


def test_append_accepts_dict_records(tmp_path):
    p = tmp_path / "l.jsonl"
    led.append({"kind": "eval", "task_id": "x"}, p)
    led.append(led.Usage(model="m", transport="converse"), p)
    rows = list(led.read(p))
    assert rows[0] == {"kind": "eval", "task_id": "x"} and rows[1]["model"] == "m"
