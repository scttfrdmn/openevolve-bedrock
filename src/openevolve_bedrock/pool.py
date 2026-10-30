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
Run an OpenEvolve evaluation on a spore.host ``spawn pool`` instead of locally.

A problem's ``evaluator.py`` keeps its own ``evaluate(program_path)``, and a
thin wrapper calls ``SpawnPool.evaluate`` to run that evaluation on a pool
worker:

1. ``stage_bundle(dir)`` tars the problem directory once (content-addressed,
   so an unchanged directory is uploaded once) to ``<staging>/bundles/``.
2. ``evaluate`` uploads the candidate to ``<staging>/candidates/<task>/``,
   ``spawn pool submit``s a TaskSpec that unpacks the bundle on the worker and
   runs ``argv`` there, then polls ``<staging>/records/<task>/completion.json``
   (the spec's ``results_prefix``) and reads ``metrics.json`` back from
   ``<staging>/results/<task>/``.
3. Each evaluation appends a ``kind: "eval"`` line to the ledger, with spawn's
   ``cost_estimate`` for the task, so ``oe-bedrock cost`` reports evaluation $
   next to LLM $.

The pool itself is created and drained with spawn's CLI around the run
(``spawn pool create --run-id R ... --s3-read/--s3-write <staging bucket>``,
``spawn pool drain --run-id R``); see examples/kissat/README.md. Configuration
is by environment, as for the rest of this package:

    OE_BEDROCK_POOL_RUN_ID   pool run id (required)
    OE_BEDROCK_POOL_STAGING  s3://bucket/prefix the workers can read and write (required)
    OE_BEDROCK_POOL_REGION   region of the pool (default: OE_BEDROCK_REGION / AWS_REGION)
    OE_BEDROCK_POOL_TIMEOUT  seconds to wait for one evaluation (default 3600)
    OE_BEDROCK_POOL_POLL     seconds between completion checks (default 10)
    OE_BEDROCK_PROFILE       boto3 / spawn profile (else AWS_PROFILE)
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import shlex
import subprocess
import tarfile
import tempfile
import time
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from openevolve_bedrock import ledger as _ledger

logger = logging.getLogger(__name__)

EXCLUDE = (".cache", "__pycache__", "openevolve_output", ".pytest_cache")


def _split_s3(uri: str) -> Tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"not an s3:// URI: {uri!r}")
    bucket, _, key = uri[5:].partition("/")
    return bucket, key.strip("/")


def _bundle_bytes(directory: Path, exclude: Iterable[str] = EXCLUDE) -> bytes:
    """Deterministic tar.gz of a directory: sorted, fixed mtimes and owners."""
    excluded = set(exclude)
    buf = io.BytesIO()
    import gzip

    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz, \
            tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for path in sorted(directory.rglob("*")):
            rel = path.relative_to(directory)
            if any(part in excluded for part in rel.parts) or not path.is_file():
                continue
            info = tar.gettarinfo(str(path), arcname=str(rel))
            info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, "", ""
            with open(path, "rb") as f:
                tar.addfile(info, f)
    return buf.getvalue()


class SpawnPool:
    def __init__(self, run_id: Optional[str] = None, staging: Optional[str] = None,
                 region: Optional[str] = None, profile: Optional[str] = None,
                 timeout_s: Optional[float] = None, poll_s: Optional[float] = None,
                 s3=None) -> None:
        self.run_id = run_id or os.environ.get("OE_BEDROCK_POOL_RUN_ID")
        self.staging = (staging or os.environ.get("OE_BEDROCK_POOL_STAGING") or "").rstrip("/")
        if not self.run_id or not self.staging:
            raise ValueError("openevolve-bedrock pool: set OE_BEDROCK_POOL_RUN_ID and "
                             "OE_BEDROCK_POOL_STAGING (s3://bucket/prefix)")
        self.bucket, self.prefix = _split_s3(self.staging)
        self.region = (region or os.environ.get("OE_BEDROCK_POOL_REGION")
                       or os.environ.get("OE_BEDROCK_REGION") or os.environ.get("AWS_REGION"))
        self.profile = profile or os.environ.get("OE_BEDROCK_PROFILE") or os.environ.get("AWS_PROFILE")
        self.timeout_s = float(timeout_s or os.environ.get("OE_BEDROCK_POOL_TIMEOUT", 3600))
        self.poll_s = float(poll_s or os.environ.get("OE_BEDROCK_POOL_POLL", 10))
        self._s3 = s3
        self._bundles: Dict[str, str] = {}

    @property
    def s3(self):
        if self._s3 is None:
            import boto3

            self._s3 = boto3.Session(profile_name=self.profile, region_name=self.region).client("s3")
        return self._s3

    def _key(self, *parts: str) -> str:
        return "/".join(p for p in (self.prefix, *parts) if p)

    def _exists(self, key: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception as e:  # botocore ClientError 404, or a fake's KeyError
            code = getattr(e, "response", {}).get("Error", {}).get("Code")
            if code in (None, "404", "NoSuchKey", "NotFound"):
                return False
            raise

    def _get_json(self, key: str) -> Optional[dict]:
        if not self._exists(key):
            return None
        body = self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        return json.loads(body)

    def stage_bundle(self, directory, exclude: Iterable[str] = EXCLUDE) -> str:
        """Upload a directory as a content-addressed tar.gz; return its s3:// URI."""
        directory = Path(directory).resolve()
        data = _bundle_bytes(directory, exclude)
        digest = hashlib.sha256(data).hexdigest()
        if digest not in self._bundles:
            key = self._key("bundles", f"{digest}.tar.gz")
            if not self._exists(key):
                self.s3.put_object(Bucket=self.bucket, Key=key, Body=data)
                logger.info("staged %s (%d bytes) to s3://%s/%s", directory, len(data), self.bucket, key)
            self._bundles[digest] = f"s3://{self.bucket}/{key}"
        return self._bundles[digest]

    def task_spec(self, task_id: str, bundle_uri: str, program_name: str, argv: List[str],
                  env: Optional[Dict[str, str]] = None, gpus: int = 0,
                  container: Optional[str] = None) -> dict:
        """The TaskSpec for one evaluation. ``{program}`` and ``{out}`` in argv are
        replaced by the candidate's path and the output dir on the worker."""
        program = f"/work/candidate/{program_name}"
        args = [a.replace("{program}", program).replace("{out}", "/work/out") for a in argv]
        script = ("set -e; mkdir -p /work/bundle /work/out; "
                  "tar -xzf /work/bundle.tar.gz -C /work/bundle; cd /work/bundle; "
                  f"exec {shlex.join(args)}")
        spec = {
            "task_id": task_id,
            "command": ["bash", "-c", script],
            "inputs": [
                {"source": bundle_uri, "destination": "/work/bundle.tar.gz"},
                {"source": f"s3://{self.bucket}/{self._key('candidates', task_id, program_name)}",
                 "destination": program},
            ],
            "outputs": [{"source": "/work/out/",
                         "destination": f"s3://{self.bucket}/{self._key('results', task_id)}/"}],
            "results_prefix": f"s3://{self.bucket}/{self._key('records')}",
            "env": dict(env or {}),
            # Ignored in a pool (the pool's flags decide), but required to validate.
            "lifecycle": {"ttl": f"{max(1, int(self.timeout_s // 60))}m"},
        }
        if gpus:
            spec["resources"] = {"gpus": gpus}
        if container:
            spec["container"] = container
        return spec

    def _submit(self, spec: dict) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(spec, f)
            path = f.name
        cmd = ["spawn", "pool", "submit", "--run-id", self.run_id, "--spec", path]
        if self.region:
            cmd += ["--region", self.region]
        if self.profile:
            cmd += ["--profile", self.profile]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        finally:
            os.unlink(path)
        if p.returncode != 0:
            raise RuntimeError(f"spawn pool submit failed: {(p.stderr or p.stdout).strip()[-800:]}")

    def evaluate(self, bundle_uri: str, program_path: str, argv: List[str],
                 env: Optional[Dict[str, str]] = None, gpus: int = 0,
                 container: Optional[str] = None) -> dict:
        """Run argv on a pool worker against the candidate; return its metrics.json.

        A task that fails, times out, or writes no metrics scores 0 with an
        ``error``, as a local evaluator failure would.
        """
        task_id = f"{self.run_id}-{uuid.uuid4().hex[:12]}"
        name = Path(program_path).name
        self.s3.upload_file(str(program_path), self.bucket, self._key("candidates", task_id, name))
        spec = self.task_spec(task_id, bundle_uri, name, argv, env, gpus, container)
        t0 = time.monotonic()
        self._submit(spec)
        record_key = self._key("records", task_id, "completion.json")
        record = None
        while time.monotonic() - t0 < self.timeout_s:
            record = self._get_json(record_key)
            if record is not None:
                break
            time.sleep(self.poll_s)
        eval_s = time.monotonic() - t0
        state = (record or {}).get("state", "timeout")
        _ledger.append({
            "ts": time.time(), "kind": "eval", "backend": "spawn-pool", "run_id": self.run_id,
            "task_id": task_id, "state": state, "exit_code": (record or {}).get("exit_code"),
            "eval_s": round(eval_s, 3), "cost_usd": (record or {}).get("cost_estimate"),
            "retry_class": (record or {}).get("retry_class"),
        })
        if record is None:
            return {"combined_score": 0.0,
                    "error": f"pool task {task_id} gave no completion record in {self.timeout_s:.0f} s"}
        metrics = self._get_json(self._key("results", task_id, "metrics.json"))
        if state != "completed" or metrics is None:
            reason = record.get("retry_class") or record.get("terminal_reason") or f"exit {record.get('exit_code')}"
            return {"combined_score": 0.0,
                    "error": f"pool task {task_id} {state} ({reason}); log: {record.get('logs')}"}
        return metrics
