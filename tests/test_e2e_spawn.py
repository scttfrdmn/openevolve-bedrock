# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""
Offline end-to-end: a real OpenEvolve run (process-parallel, *spawn* start
method as on macOS) against a fake Mantle server. Proves the provider is
registered inside spawned workers, that retry absorbs 429s, that the
ledger and cost report line up with the run, and that model draws are seeded
per iteration rather than identically in every worker (openevolve#554).
"""
import hashlib
import json
import os
import random
import socket
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import pytest
import yaml

HERE = Path(__file__).resolve().parent
EXAMPLE = HERE.parent / "examples" / "function_minimization"

DRIVER = """
import multiprocessing as mp, sys
mp.set_start_method("spawn", force=True)
from openevolve_bedrock.cli import main
if __name__ == "__main__":
    sys.exit(main())
"""


def _expected_counts(cfg: dict) -> Counter:
    """Model calls per name if iteration i draws from Random(md5(f"{llm_seed}:{i}"))."""
    base = str(cfg["random_seed"]).encode()
    llm_seed = int(hashlib.md5(base + b"llm", usedforsecurity=False).hexdigest()[:8], 16) % (2**31)
    models = cfg["llm"]["models"]
    total = sum(m["weight"] for m in models)
    weights = [m["weight"] / total for m in models]
    counts = Counter()
    for i in range(1, cfg["max_iterations"] + 1):
        seed = int(hashlib.md5(f"{llm_seed}:{i}".encode(), usedforsecurity=False).hexdigest()[:8], 16)
        idx = random.Random(seed).choices(range(len(models)), weights=weights, k=1)[0]
        counts[models[idx]["name"]] += 1
    return counts


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.skipif(not EXAMPLE.exists(), reason="examples not present")
def test_openevolve_spawn_workers_use_bedrock_provider(tmp_path):
    pytest.importorskip("numpy")
    port = _free_port()
    server = subprocess.Popen(
        [sys.executable, str(HERE / "fake_mantle.py"), str(port), str(EXAMPLE / "initial_program.py")]
    )
    try:
        # wait for the port
        for _ in range(50):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                break
            except OSError:
                time.sleep(0.1)
        cfg = (HERE / "e2e_config.yaml").read_text().replace("__PORT__", str(port))
        (tmp_path / "config.yaml").write_text(cfg)
        (tmp_path / "driver.py").write_text(DRIVER)
        cfg = yaml.safe_load(cfg)
        n = cfg["max_iterations"]
        expected = _expected_counts(cfg)
        assert len(expected) == 2, "seed should exercise both models"

        runs = []
        for run in ("a", "b"):
            out = tmp_path / f"out_{run}"
            env = dict(os.environ, AWS_BEARER_TOKEN_BEDROCK="fake", OE_BEDROCK_MAX_ATTEMPTS="6")
            env.pop("OE_BEDROCK_LEDGER", None)  # let `run` default it into the output dir
            proc = subprocess.run(
                [sys.executable, str(tmp_path / "driver.py"), "run",
                 str(EXAMPLE / "initial_program.py"), str(EXAMPLE / "evaluator.py"),
                 "--config", str(tmp_path / "config.yaml"), "--output", str(out)],
                env=env, cwd=str(tmp_path), capture_output=True, text=True, timeout=600,
            )
            assert proc.returncode == 0, proc.stderr[-3000:]
            ledger = out / "bedrock_ledger.jsonl"
            assert ledger.is_file(), "ledger not written by workers"
            recs = [json.loads(l) for l in ledger.read_text().splitlines() if l.strip()]
            assert len(recs) == n, f"expected one ledger line per iteration, got {len(recs)}"
            assert {r["transport"] for r in recs} == {"openai"}
            assert all(r["input_tokens"] > 0 for r in recs)
            assert (out / "best" / "best_program_info.json").is_file()
            assert (out / "checkpoints" / f"checkpoint_{n}" / "best_program_info.json").is_file()
            assert "Initialized Bedrock LLM" in proc.stderr + proc.stdout
            runs.append(Counter(r["model"] for r in recs))

        # Counts do not depend on which worker ran which iteration, so they are exact.
        assert runs[0] == expected, f"realized {dict(runs[0])}, expected {dict(expected)}"
        assert runs[1] == runs[0], "same seed must give the same realized mix"
    finally:
        server.terminate()
        server.wait(timeout=10)


def _run_with_fake_mantle(tmp_path, extra_args, pricing_price=0.005):
    """One spawn-mode `oe-bedrock run` of e2e_config.yaml; returns (proc, out, cfg, pricing path)."""
    port = _free_port()
    server = subprocess.Popen(
        [sys.executable, str(HERE / "fake_mantle.py"), str(port), str(EXAMPLE / "initial_program.py")]
    )
    try:
        for _ in range(50):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                break
            except OSError:
                time.sleep(0.1)
        cfg = (HERE / "e2e_config.yaml").read_text().replace("__PORT__", str(port))
        (tmp_path / "config.yaml").write_text(cfg)
        (tmp_path / "driver.py").write_text(DRIVER)
        cfg = yaml.safe_load(cfg)
        pricing = tmp_path / "pricing.yaml"
        pricing.write_text("models:\n" + "".join(
            f"  {m['name']}: {{input: {pricing_price}, output: {pricing_price}}}\n"
            for m in cfg["llm"]["models"]))
        out = tmp_path / "out"
        env = dict(os.environ, AWS_BEARER_TOKEN_BEDROCK="fake", OE_BEDROCK_MAX_ATTEMPTS="6")
        env.pop("OE_BEDROCK_LEDGER", None)
        proc = subprocess.run(
            [sys.executable, str(tmp_path / "driver.py"), "run",
             str(EXAMPLE / "initial_program.py"), str(EXAMPLE / "evaluator.py"),
             "--config", str(tmp_path / "config.yaml"), "--output", str(out), *extra_args],
            env=env, cwd=str(tmp_path), capture_output=True, text=True, timeout=600,
        )
        assert proc.returncode == 0, proc.stderr[-3000:]
        return proc, out, cfg, pricing
    finally:
        server.terminate()
        server.wait(timeout=10)


@pytest.mark.skipif(not EXAMPLE.exists(), reason="examples not present")
def test_budget_stops_run_and_checkpoints_last_iteration(tmp_path):
    """--budget-usd stops a spawn-mode run within one batch of in-flight calls of
    the budget; the last checkpoint is where it stopped, not checkpoint_40."""
    pytest.importorskip("numpy")
    from openevolve_bedrock import ledger as led

    budget = 0.08  # ~$0.01 per call at the fake server's ~1-2K prompt tokens
    pricing_args = ["--budget-usd", str(budget)]
    proc, out, cfg, pricing = _run_with_fake_mantle(tmp_path, pricing_args + ["--pricing", str(tmp_path / "pricing.yaml")])
    recs = list(led.read(out / "bedrock_ledger.jsonl"))
    prices = led.load_pricing(pricing)
    spent = led.summarize(recs, prices).total_usd
    per_call = max(led.summarize([r], prices).total_usd for r in recs)
    in_flight = 2 * cfg["evaluator"]["parallel_evaluations"]
    assert budget <= spent < budget + in_flight * per_call, (spent, len(recs))
    assert "Budget $0.0800 reached" in proc.stderr + proc.stdout
    series = led.checkpoint_series(out, "combined_score")
    assert series[-1].iteration == len(recs) < 40   # last checkpoint = where it stopped
    assert not (out / "checkpoints" / "checkpoint_40").exists()
    assert (out / "best" / "best_program_info.json").is_file()


@pytest.mark.skipif(not EXAMPLE.exists(), reason="examples not present")
def test_target_score_stop_checkpoints_last_iteration(tmp_path):
    """--target-score is OpenEvolve's; the wrapper fixes its final checkpoint label."""
    pytest.importorskip("numpy")
    from openevolve_bedrock import ledger as led

    proc, out, cfg, _ = _run_with_fake_mantle(tmp_path, ["--target-score", "1.0"])
    assert "Target score 1.0 reached" in proc.stderr + proc.stdout
    series = led.checkpoint_series(out, "combined_score")
    assert series and series[-1].iteration < 40 and series[-1].metric_value >= 1.0
    assert not (out / "checkpoints" / "checkpoint_40").exists()
    assert (out / "best" / "best_program_info.json").is_file()
