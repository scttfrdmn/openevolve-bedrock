# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""pool.SpawnPool against a fake S3 and a fake `spawn pool submit`."""
import io
import json
import os
import tarfile
import gzip
from pathlib import Path
from types import SimpleNamespace

import pytest

from openevolve_bedrock import ledger as led
from openevolve_bedrock import pool as pool_mod


class FakeS3:
    def __init__(self):
        self.objects = {}

    def head_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            err = Exception("404")
            err.response = {"Error": {"Code": "404"}}
            raise err
        return {}

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def put_object(self, Bucket, Key, Body):
        self.objects[(Bucket, Key)] = Body

    def upload_file(self, filename, bucket, key):
        self.objects[(bucket, key)] = Path(filename).read_bytes()


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("OE_BEDROCK_LEDGER", str(tmp_path / "ledger.jsonl"))
    monkeypatch.setattr(pool_mod.time, "sleep", lambda s: None)
    return tmp_path


def _pool(s3, **kw):
    return pool_mod.SpawnPool(run_id="r1", staging="s3://bkt/oe", region="us-west-2",
                              profile="aws", timeout_s=kw.pop("timeout_s", 60), poll_s=1, s3=s3, **kw)


def _fake_worker(monkeypatch, s3, state="completed", metrics=None, cost=0.04, write_record=True,
                 started=True):
    """`spawn pool submit` that 'runs' the task: writes its record and metrics."""
    calls = []

    def run(cmd, capture_output, text, timeout):
        spec = json.loads(Path(cmd[cmd.index("--spec") + 1]).read_text())
        calls.append((cmd, spec))
        tid = spec["task_id"]
        if started:  # what the spec's first command does on a worker
            s3.put_object("bkt", f"oe/started/{tid}", b"1793670627\n")
        rec_bucket, rec_prefix = pool_mod._split_s3(spec["results_prefix"])
        if write_record:
            s3.put_object(rec_bucket, f"{rec_prefix}/{tid}/completion.json",
                          json.dumps({"task_id": tid, "state": state, "exit_code": 0 if state == "completed" else 1,
                                      "cost_estimate": cost, "logs": f"s3://x/{tid}/command.log",
                                      "started_at": "2026-11-03T01:50:27Z", "ended_at": "2026-11-03T01:50:54Z",
                                      "instance_type": "c8g.8xlarge"}).encode())
        if metrics is not None:
            out_bucket, out_prefix = pool_mod._split_s3(spec["outputs"][0]["destination"])
            s3.put_object(out_bucket, f"{out_prefix}/metrics.json", json.dumps(metrics).encode())
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pool_mod.subprocess, "run", run)
    return calls


def test_requires_run_id_and_staging(monkeypatch):
    monkeypatch.delenv("OE_BEDROCK_POOL_RUN_ID", raising=False)
    monkeypatch.delenv("OE_BEDROCK_POOL_STAGING", raising=False)
    with pytest.raises(ValueError, match="OE_BEDROCK_POOL_RUN_ID"):
        pool_mod.SpawnPool()


def test_bundle_is_content_addressed_and_skips_excluded(env, tmp_path):
    d = tmp_path / "problem"
    (d / ".cache").mkdir(parents=True)
    (d / "evaluator.py").write_text("x = 1\n")
    (d / ".cache" / "big.bin").write_bytes(b"0" * 1000)
    s3 = FakeS3()
    p = _pool(s3)
    uri = p.stage_bundle(d)
    os.utime(d / "evaluator.py", (1, 1))                 # mtime doesn't change the content hash
    assert _pool(s3).stage_bundle(d) == uri
    (d / "evaluator.py").write_text("x = 2\n")
    assert _pool(s3).stage_bundle(d) != uri
    data = s3.objects[pool_mod._split_s3(uri)]
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(data))) as t:
        assert t.getnames() == ["evaluator.py"]


def test_evaluate_round_trip_and_ledger(env, tmp_path, monkeypatch):
    s3 = FakeS3()
    prog = tmp_path / "cand.c"
    prog.write_text("int x;\n")
    calls = _fake_worker(monkeypatch, s3, metrics={"combined_score": 1.25, "solved": 24})
    p = _pool(s3)
    m = p.evaluate("s3://bkt/oe/bundles/abc.tar.gz", str(prog),
                   ["python3", "remote.py", "stage2", "{program}", "{out}/metrics.json"],
                   env={"KISSAT_EVAL_JOBS": "24"})
    assert m == {"combined_score": 1.25, "solved": 24}
    cmd, spec = calls[0]
    assert cmd[:6] == ["spawn", "pool", "submit", "--run-id", "r1", "--spec"]
    assert cmd[7:] == ["--region", "us-west-2", "--profile", "aws"]
    assert spec["task_id"].startswith("r1-") and spec["env"] == {"KISSAT_EVAL_JOBS": "24"}
    assert spec["command"][2].endswith(
        "exec python3 remote.py stage2 /work/candidate/cand.c /work/out/metrics.json")
    assert spec["command"][2].startswith(
        f"date -u +%s | aws s3 cp --quiet --region us-west-2 - s3://bkt/oe/started/{spec['task_id']} || true; set -e;")
    assert spec["inputs"][1] == {"source": f"s3://bkt/oe/candidates/{spec['task_id']}/cand.c",
                                 "destination": "/work/candidate/cand.c"}
    assert s3.objects[("bkt", f"oe/candidates/{spec['task_id']}/cand.c")] == b"int x;\n"
    assert spec["results_prefix"] == "s3://bkt/oe/records"
    rec = list(led.read(env / "ledger.jsonl"))[0]
    assert rec["kind"] == "eval" and rec["state"] == "completed" and rec["cost_usd"] == 0.04
    assert rec["task_id"] == spec["task_id"]
    assert rec["run_s"] == 27.0 and rec["instance_type"] == "c8g.8xlarge"
    assert rec["started_at"] == 1793670627.0 and rec["ended_at"] == 1793670654.0
    assert rec["wait_s"] == pytest.approx(rec["started_at"] - rec["submitted_at"], abs=1e-3)


def test_failed_task_scores_zero_with_reason(env, tmp_path, monkeypatch):
    s3 = FakeS3()
    prog = tmp_path / "cand.c"
    prog.write_text("x")
    _fake_worker(monkeypatch, s3, state="failed", metrics=None)
    m = _pool(s3).evaluate("s3://bkt/oe/b.tgz", str(prog), ["true"])
    assert m["combined_score"] == 0.0 and "failed" in m["error"]


def test_no_record_times_out(env, tmp_path, monkeypatch):
    s3 = FakeS3()
    prog = tmp_path / "cand.c"
    prog.write_text("x")
    _fake_worker(monkeypatch, s3, write_record=False)
    clock = iter(range(0, 10_000, 30))
    monkeypatch.setattr(pool_mod.time, "monotonic", lambda: next(clock))
    m = _pool(s3, timeout_s=100).evaluate("s3://bkt/oe/b.tgz", str(prog), ["true"])
    assert m["combined_score"] == 0.0 and "no completion record in 100 s of running" in m["error"]
    rec = list(led.read(env / "ledger.jsonl"))[0]
    assert rec["state"] == "timeout" and rec["run_s"] is None and rec["wait_s"] is None


def _delayed_record(monkeypatch, s3, start_after, finish_after):
    """A worker that takes the task after `start_after` polls and finishes after `finish_after`."""
    polls = {"n": 0}
    calls = _fake_worker(monkeypatch, s3, write_record=False, started=False, metrics={"combined_score": 1.0})
    get = s3.get_object
    head = s3.head_object

    def head_object(Bucket, Key):
        if Key.endswith("completion.json") or "/started/" in Key:
            polls["n"] += Key.endswith("completion.json")
            tid = calls[0][1]["task_id"]
            if polls["n"] >= start_after:
                s3.objects.setdefault(("bkt", f"oe/started/{tid}"), b"1")
            if polls["n"] >= finish_after:
                s3.objects.setdefault(("bkt", f"oe/records/{tid}/completion.json"),
                                      json.dumps({"task_id": tid, "state": "completed", "exit_code": 0}).encode())
        return head(Bucket, Key)

    s3.head_object = head_object
    return polls


def test_queue_time_does_not_count_toward_the_timeout(env, tmp_path, monkeypatch):
    s3 = FakeS3()
    prog = tmp_path / "cand.c"
    prog.write_text("x")
    _delayed_record(monkeypatch, s3, start_after=10, finish_after=12)   # queued ~300 s, runs ~60 s
    clock = iter(range(0, 10_000, 30))
    monkeypatch.setattr(pool_mod.time, "monotonic", lambda: next(clock))
    m = _pool(s3, timeout_s=100).evaluate("s3://bkt/oe/b.tgz", str(prog), ["true"])
    assert m == {"combined_score": 1.0}
    assert list(led.read(env / "ledger.jsonl"))[0]["state"] == "completed"


def test_a_task_never_started_hits_the_queue_timeout(env, tmp_path, monkeypatch):
    s3 = FakeS3()
    prog = tmp_path / "cand.c"
    prog.write_text("x")
    _fake_worker(monkeypatch, s3, write_record=False, started=False)
    clock = iter(range(0, 100_000, 30))
    monkeypatch.setattr(pool_mod.time, "monotonic", lambda: next(clock))
    m = _pool(s3, timeout_s=100, queue_timeout_s=1000).evaluate("s3://bkt/oe/b.tgz", str(prog), ["true"])
    assert "not started by a worker in 1000 s" in m["error"]
    assert list(led.read(env / "ledger.jsonl"))[0]["state"] == "queue_timeout"


def test_submit_failure_raises(env, tmp_path, monkeypatch):
    s3 = FakeS3()
    prog = tmp_path / "cand.c"
    prog.write_text("x")
    monkeypatch.setattr(pool_mod.subprocess, "run",
                        lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="no queue"))
    with pytest.raises(RuntimeError, match="no queue"):
        _pool(s3).evaluate("s3://bkt/oe/b.tgz", str(prog), ["true"])
