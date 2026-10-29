# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
import os
from types import SimpleNamespace

import pytest

import openevolve_bedrock.provider as prov


class FakeConverseClient:
    """Records converse() calls; `script` is a list of responses or exceptions."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def converse_ok(text="OK", inp=10, out=2, cr=0, cw=0, stop="end_turn"):
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "usage": {"inputTokens": inp, "outputTokens": out, "totalTokens": inp + out,
                  "cacheReadInputTokens": cr, "cacheWriteInputTokens": cw},
        "stopReason": stop,
    }


def client_error(code, message="boom"):
    from botocore.exceptions import ClientError

    return ClientError({"Error": {"Code": code, "Message": message}}, "Converse")


def model_cfg(**overrides):
    base = dict(
        name="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        system_message="sys",
        temperature=0.7,
        top_p=None,
        max_tokens=512,
        timeout=30,
        retries=1,
        retry_delay=0,
        api_base="aws://us-west-2",
        api_key=None,
        random_seed=None,
        reasoning_effort=None,
        provider="bedrock",
        init_client=None,
        weight=1.0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """Fresh ledger per test, no sleeping, no cached clients, cache set reset."""
    monkeypatch.setenv("OE_BEDROCK_LEDGER", str(tmp_path / "ledger.jsonl"))
    monkeypatch.delenv("OE_BEDROCK_CACHE", raising=False)
    monkeypatch.delenv("OE_BEDROCK_THINKING_BUDGET", raising=False)
    monkeypatch.setenv("OE_BEDROCK_MAX_ATTEMPTS", "4")
    monkeypatch.setattr(prov.time, "sleep", lambda s: None)
    prov._client_cache.clear()
    prov._no_cache_models.clear()
    yield


@pytest.fixture
def fake_converse(monkeypatch):
    def _install(script):
        client = FakeConverseClient(script)
        monkeypatch.setattr(prov, "_bedrock_runtime_client", lambda *a, **k: client)
        return client

    return _install
