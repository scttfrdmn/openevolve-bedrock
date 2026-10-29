# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import openevolve_bedrock.provider as prov
from openevolve_bedrock.provider import BedrockLLM

from conftest import client_error, converse_ok, model_cfg


def run(coro):
    return asyncio.run(coro)


def read_ledger():
    p = Path(os.environ["OE_BEDROCK_LEDGER"])
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


# ---------------------------------------------------------------- converse


def test_converse_request_shape_and_usage(fake_converse):
    client = fake_converse([converse_ok("hello", inp=100, out=20, cr=60, cw=0)])
    llm = BedrockLLM(model_cfg())
    text = run(llm.generate_with_context("SYSTEM", [{"role": "user", "content": "hi"}]))
    assert text == "hello"

    req = client.calls[0]
    assert req["modelId"] == "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    assert req["messages"] == [{"role": "user", "content": [{"text": "hi"}]}]
    assert req["system"] == [{"text": "SYSTEM"}, {"cachePoint": {"type": "default"}}]
    assert req["inferenceConfig"] == {"maxTokens": 512, "temperature": 0.7}
    assert "topP" not in req["inferenceConfig"]          # top_p None never sent
    assert "additionalModelRequestFields" not in req

    assert llm.last_usage["prompt_tokens"] == 100
    assert llm.last_usage["completion_tokens"] == 20
    assert llm.last_usage["cache_read_tokens"] == 60

    rec = read_ledger()[0]
    assert rec["model"].startswith("us.anthropic")
    assert rec["transport"] == "converse"
    assert (rec["input_tokens"], rec["output_tokens"], rec["cache_read_tokens"]) == (100, 20, 60)
    assert rec["attempts"] == 1 and rec["throttled"] == 0
    assert rec["stop_reason"] == "end_turn"


def test_top_p_forwarded_when_set(fake_converse):
    client = fake_converse([converse_ok()])
    llm = BedrockLLM(model_cfg(top_p=0.95))
    run(llm.generate("x"))
    assert client.calls[0]["inferenceConfig"]["topP"] == 0.95


def test_seed_is_ignored(fake_converse):
    client = fake_converse([converse_ok()])
    llm = BedrockLLM(model_cfg(random_seed=7))
    run(llm.generate("x", seed=7))
    assert "seed" not in json.dumps(client.calls[0])


def test_cache_off_env(fake_converse, monkeypatch):
    monkeypatch.setenv("OE_BEDROCK_CACHE", "off")
    client = fake_converse([converse_ok()])
    run(BedrockLLM(model_cfg()).generate("x"))
    assert client.calls[0]["system"] == [{"text": "sys"}]


def test_cache_point_rejected_then_disabled(fake_converse):
    client = fake_converse([
        client_error("ValidationException", "This model doesn't support cachePoint"),
        converse_ok(),
        converse_ok(),
    ])
    llm = BedrockLLM(model_cfg())
    run(llm.generate("x"))
    assert "cachePoint" in json.dumps(client.calls[0])
    assert "cachePoint" not in json.dumps(client.calls[1])
    assert read_ledger()[0]["attempts"] == 1   # the fallback retry is not an attempt

    run(llm.generate("y"))                      # second call skips the cache point up front
    assert len(client.calls) == 3
    assert "cachePoint" not in json.dumps(client.calls[2])


def test_throttle_retry_counts(fake_converse):
    client = fake_converse([
        client_error("ThrottlingException", "slow down"),
        client_error("ServiceUnavailableException", "busy"),
        converse_ok(),
    ])
    run(BedrockLLM(model_cfg()).generate("x"))
    assert len(client.calls) == 3
    rec = read_ledger()[0]
    assert rec["attempts"] == 3 and rec["throttled"] == 2


def test_throttle_budget_exhausted_raises(fake_converse):
    from botocore.exceptions import ClientError

    fake_converse([client_error("ThrottlingException")] * 4)
    with pytest.raises(ClientError):
        run(BedrockLLM(model_cfg()).generate("x"))


def test_non_retryable_error_raises_immediately(fake_converse):
    from botocore.exceptions import ClientError

    client = fake_converse([client_error("AccessDeniedException", "no"), converse_ok()])
    with pytest.raises(ClientError):
        run(BedrockLLM(model_cfg()).generate("x"))
    assert len(client.calls) == 1


def test_anthropic_reasoning_effort_maps_to_thinking(fake_converse):
    client = fake_converse([converse_ok()])
    run(BedrockLLM(model_cfg(reasoning_effort="high", max_tokens=2000)).generate("x"))
    req = client.calls[0]
    assert req["additionalModelRequestFields"] == {"thinking": {"type": "enabled", "budget_tokens": 16384}}
    assert req["inferenceConfig"] == {"maxTokens": 16384 + 1024}   # > budget, no temperature/topP


def test_thinking_budget_env(fake_converse, monkeypatch):
    monkeypatch.setenv("OE_BEDROCK_THINKING_BUDGET", "2048")
    client = fake_converse([converse_ok()])
    run(BedrockLLM(model_cfg()).generate("x"))
    assert client.calls[0]["additionalModelRequestFields"]["thinking"]["budget_tokens"] == 2048


def test_openai_vendor_reasoning_effort(fake_converse):
    client = fake_converse([converse_ok()])
    run(BedrockLLM(model_cfg(name="openai.gpt-oss-120b-1:0", reasoning_effort="low")).generate("x"))
    req = client.calls[0]
    assert req["additionalModelRequestFields"] == {"reasoning_effort": "low"}
    assert req["inferenceConfig"]["temperature"] == 0.7


def test_reasoning_text_blocks_are_skipped(fake_converse):
    resp = converse_ok("answer")
    resp["output"]["message"]["content"].insert(0, {"reasoningContent": {"reasoningText": {"text": "thinking..."}}})
    fake_converse([resp])
    assert run(BedrockLLM(model_cfg()).generate("x")) == "answer"


def test_region_resolution(monkeypatch):
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    monkeypatch.delenv("OE_BEDROCK_REGION", raising=False)
    assert BedrockLLM(model_cfg(api_base="aws://eu-central-1")).transport.region == "eu-central-1"
    monkeypatch.setenv("OE_BEDROCK_REGION", "us-east-2")
    assert BedrockLLM(model_cfg(api_base="aws://")).transport.region == "us-east-2"
    assert BedrockLLM(model_cfg(api_base=None)).transport.region == "us-east-2"
    monkeypatch.delenv("OE_BEDROCK_REGION")
    with pytest.raises(ValueError):
        BedrockLLM(model_cfg(api_base=None))


# ---------------------------------------------------------------- openai-compatible


class _FakeOpenAI:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _chat_ok(text="hi", pt=50, ct=5, cached=0):
    details = SimpleNamespace(cached_tokens=cached)
    usage = SimpleNamespace(prompt_tokens=pt, completion_tokens=ct, prompt_tokens_details=details)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")],
        usage=usage,
    )


@pytest.fixture
def fake_openai(monkeypatch):
    def _install(script):
        fake = _FakeOpenAI(script)
        import openai as _openai

        monkeypatch.setattr(_openai, "OpenAI", lambda **k: fake)
        return fake

    return _install


def test_openai_transport_request(fake_openai, monkeypatch):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "k")
    fake = fake_openai([_chat_ok(pt=70, ct=7, cached=30)])
    llm = BedrockLLM(model_cfg(api_base="https://bedrock-mantle.us-east-1.api.aws/v1", random_seed=3))
    assert llm.transport.name == "openai"
    text = run(llm.generate_with_context("S", [{"role": "user", "content": "u"}], seed=3))
    assert text == "hi"
    req = fake.calls[0]
    assert req["messages"][0] == {"role": "system", "content": "S"}
    assert "seed" not in req and "top_p" not in req
    assert req["temperature"] == 0.7 and req["max_tokens"] == 512 and req["stream"] is False
    rec = read_ledger()[0]
    assert rec["transport"] == "openai" and rec["cache_read_tokens"] == 30
    assert rec["input_tokens"] == 70 and rec["output_tokens"] == 7


def test_openai_transport_429_retry(fake_openai, monkeypatch):
    import openai as _openai

    try:
        import httpx
    except ModuleNotFoundError:  # openai>=3 depends on httpx2
        import httpx2 as httpx

    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "k")
    resp = httpx.Response(429, request=httpx.Request("POST", "https://x"))
    err = _openai.RateLimitError("rate", response=resp, body=None)
    fake = fake_openai([err, _chat_ok()])
    run(BedrockLLM(model_cfg(api_base="https://bedrock-mantle.us-east-1.api.aws/v1")).generate("x"))
    assert len(fake.calls) == 2
    assert read_ledger()[0]["throttled"] == 1


def test_openai_transport_requires_key(monkeypatch):
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        BedrockLLM(model_cfg(api_base="https://bedrock-mantle.us-east-1.api.aws/v1"))


# ---------------------------------------------------------------- registration with OpenEvolve


def test_openevolve_config_builds_bedrock_models(fake_converse):
    """provider: bedrock at the llm level propagates to every model, in the parent
    and in the worker-style reconstruction OpenEvolve does after spawn."""
    from dataclasses import asdict

    from openevolve.config import Config, LLMModelConfig
    from openevolve.llm.ensemble import LLMEnsemble

    cfg = Config.from_dict({
        "llm": {
            "provider": "bedrock",
            "api_base": "aws://us-west-2",
            "models": [
                {"name": "us.anthropic.claude-haiku-4-5-20251001-v1:0", "weight": 0.8},
                {"name": "us.anthropic.claude-sonnet-4-5-20250929-v1:0", "weight": 0.2},
            ],
        }
    })
    ens = LLMEnsemble(cfg.llm.models)
    assert all(isinstance(m, BedrockLLM) for m in ens.models)
    assert [m.transport.region for m in ens.models] == ["us-west-2", "us-west-2"]

    # worker path: dicts -> LLMModelConfig -> ensemble
    worker_models = [LLMModelConfig(**asdict(m)) for m in cfg.llm.models]
    ens2 = LLMEnsemble(worker_models)
    assert all(isinstance(m, BedrockLLM) for m in ens2.models)

    client = fake_converse([converse_ok("evolved")])
    assert run(ens2.generate_with_context("S", [{"role": "user", "content": "u"}])) == "evolved"
    assert ens2.last_usage["prompt_tokens"] == 10
    assert client.calls[0]["modelId"].startswith("us.anthropic")


def test_register_is_idempotent():
    from openevolve.llm import ensemble

    prov.register()
    prov.register()
    assert ensemble._PROVIDER_REGISTRY["bedrock"] is not None


# ---------------------------------------------------------------- per-iteration seeding


def test_iteration_worker_is_patched_and_pickles_by_reference():
    import pickle

    from openevolve import process_parallel as pp

    prov.register()
    assert pp._run_iteration_worker is prov._seeded_run_iteration_worker
    assert prov._original_run_iteration_worker is not prov._seeded_run_iteration_worker
    assert b"openevolve_bedrock.provider" in pickle.dumps(pp._run_iteration_worker)


def _fake_worker(monkeypatch, llm_seed):
    import random

    from openevolve import process_parallel as pp

    ens = SimpleNamespace(random_state=random.Random(7))
    cfg = SimpleNamespace(llm=SimpleNamespace(models=[SimpleNamespace(random_seed=llm_seed)]))
    monkeypatch.setattr(pp, "_lazy_init_worker_components", lambda: None)
    monkeypatch.setattr(pp, "_worker_llm_ensemble", ens, raising=False)
    monkeypatch.setattr(pp, "_worker_config", cfg, raising=False)
    seen = []
    monkeypatch.setattr(
        prov, "_original_run_iteration_worker",
        lambda it, *a, **k: seen.append((it, ens.random_state.random())) or "result",
    )
    return seen


def test_seeded_worker_reseeds_per_iteration(monkeypatch):
    seen = _fake_worker(monkeypatch, llm_seed=12345)
    for it in (1, 2, 1):
        assert prov._seeded_run_iteration_worker(it, {}, "p", []) == "result"
    (_, a1), (_, a2), (_, b1) = seen
    assert a1 == b1, "same (seed, iteration) must give the same draw on any worker"
    assert a1 != a2


def test_seeded_worker_leaves_unseeded_rng_alone(monkeypatch):
    import random

    seen = _fake_worker(monkeypatch, llm_seed=None)
    prov._seeded_run_iteration_worker(1, {}, "p", [])
    assert seen[0][1] == random.Random(7).random()
