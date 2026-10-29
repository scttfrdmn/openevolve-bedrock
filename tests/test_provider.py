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


def test_anthropic_reasoning_effort_maps_to_adaptive_thinking(fake_converse):
    client = fake_converse([converse_ok()])
    run(BedrockLLM(model_cfg(reasoning_effort="high", max_tokens=2000)).generate("x"))
    req = client.calls[0]
    assert req["additionalModelRequestFields"] == {
        "thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}}
    assert req["inferenceConfig"] == {"maxTokens": 2000}   # no temperature/topP with thinking


def test_adaptive_rejected_falls_back_to_budget(fake_converse):
    # Claude Haiku 4.5's rejection, verbatim from Bedrock (2026-10-28)
    client = fake_converse([
        client_error("ValidationException", "The model returned the following errors: "
                     "adaptive thinking is not supported on this model"),
        converse_ok(), converse_ok(),
    ])
    llm = BedrockLLM(model_cfg(reasoning_effort="high", max_tokens=2000))
    run(llm.generate("x"))
    assert client.calls[1]["additionalModelRequestFields"] == {
        "thinking": {"type": "enabled", "budget_tokens": 16384}}
    assert client.calls[1]["inferenceConfig"] == {"maxTokens": 16384 + 1024}  # > budget
    assert read_ledger()[0]["attempts"] == 1
    run(llm.generate("y"))                     # remembered: budget up front
    assert len(client.calls) == 3
    assert client.calls[2]["additionalModelRequestFields"]["thinking"]["type"] == "enabled"


def test_budget_rejected_falls_back_to_adaptive(fake_converse, monkeypatch):
    # Claude 5.5's rejection of budget_tokens, verbatim from Bedrock
    monkeypatch.setenv("OE_BEDROCK_THINKING_BUDGET", "2048")
    client = fake_converse([
        client_error("ValidationException", 'The model returned the following errors: '
                     '"thinking.type.enabled" is not supported for this model. '
                     'Use "thinking.type.adaptive" and "output_config.effort" instead.'),
        converse_ok(),
    ])
    run(BedrockLLM(model_cfg(name="us.anthropic.claude-sonnet-5-5")).generate("x"))
    assert client.calls[0]["additionalModelRequestFields"]["thinking"]["budget_tokens"] == 2048
    assert client.calls[1]["additionalModelRequestFields"] == {"thinking": {"type": "adaptive"}}


def test_thinking_dropped_when_neither_shape_accepted(fake_converse):
    client = fake_converse([
        client_error("ValidationException", "adaptive thinking is not supported on this model"),
        client_error("ValidationException", "budget_tokens: thinking is not supported"),
        converse_ok(),
    ])
    run(BedrockLLM(model_cfg(reasoning_effort="low")).generate("x"))
    assert len(client.calls) == 3
    assert "additionalModelRequestFields" not in client.calls[2]
    assert client.calls[2]["inferenceConfig"]["temperature"] == 0.7  # sampling back without thinking


def test_cache_point_access_denied_is_learned(fake_converse):
    # How Bedrock rejects cachePoint on most non-Claude/Nova models (2026-10-28)
    client = fake_converse([
        client_error("AccessDeniedException", "You invoked an unsupported model or your request "
                     "did not allow prompt caching. See the documentation for more information."),
        converse_ok(),
    ])
    run(BedrockLLM(model_cfg(name="deepseek.v3.2")).generate("x"))
    assert "cachePoint" in json.dumps(client.calls[0])
    assert "cachePoint" not in json.dumps(client.calls[1])


def test_temperature_rejected_then_dropped(fake_converse):
    client = fake_converse([
        client_error("ValidationException",
                     "The model returned the following errors: `temperature` is deprecated for this model."),
        converse_ok(), converse_ok(),
    ])
    llm = BedrockLLM(model_cfg(name="us.anthropic.claude-opus-5-5", top_p=0.9))
    run(llm.generate("x"))
    assert client.calls[0]["inferenceConfig"]["temperature"] == 0.7
    assert client.calls[1]["inferenceConfig"] == {"maxTokens": 512}
    run(llm.generate("y"))
    assert client.calls[2]["inferenceConfig"] == {"maxTokens": 512}


def test_reasoning_effort_field_fallbacks(fake_converse):
    # GPT-6.1 rejects reasoning_effort and takes reasoning.effort
    client = fake_converse([
        client_error("ValidationException", "The model returned the following errors: "
                     "Unknown parameter: 'reasoning_effort'."),
        converse_ok(),
    ])
    run(BedrockLLM(model_cfg(name="us.openai.gpt-6.1-sol", reasoning_effort="low")).generate("x"))
    assert client.calls[0]["additionalModelRequestFields"] == {"reasoning_effort": "low"}
    assert client.calls[1]["additionalModelRequestFields"] == {"reasoning": {"effort": "low"}}


def test_unexplained_rejection_drops_reasoning(fake_converse):
    # DeepSeek R1 rejects reasoning_effort with a bare "Validation Error"
    client = fake_converse([
        client_error("ValidationException", "The model returned the following errors: Validation Error"),
        converse_ok(),
    ])
    run(BedrockLLM(model_cfg(name="us.deepseek.r1-v1:0", reasoning_effort="low")).generate("x"))
    assert "additionalModelRequestFields" not in client.calls[1]


def test_reasoning_effort_sent_to_any_vendor(fake_converse):
    client = fake_converse([converse_ok()])
    run(BedrockLLM(model_cfg(name="deepseek.v3.2", reasoning_effort="medium")).generate("x"))
    assert client.calls[0]["additionalModelRequestFields"] == {"reasoning_effort": "medium"}


def test_unrecognised_or_repeated_rejection_raises(fake_converse):
    from botocore.exceptions import ClientError

    client = fake_converse([client_error("ValidationException", "Malformed input request"), converse_ok()])
    with pytest.raises(ClientError):
        run(BedrockLLM(model_cfg()).generate("x"))
    assert len(client.calls) == 1

    temp = "`temperature` is deprecated for this model."
    client = fake_converse([client_error("ValidationException", temp),
                            client_error("ValidationException", temp), converse_ok()])
    with pytest.raises(ClientError):                 # learned once; no second adaptation, no loop
        run(BedrockLLM(model_cfg(name="m.x")).generate("x"))
    assert len(client.calls) == 2


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


def _chat_ok(text="hi", pt=50, ct=5, cached=0, written=0):
    details = SimpleNamespace(cached_tokens=cached, cache_write_tokens=written)
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
    assert rec["input_tokens"] == 40 and rec["output_tokens"] == 7  # 70 prompt - 30 cached


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


def test_openai_transport_drops_rejected_temperature(fake_openai, monkeypatch):
    import openai as _openai

    try:
        import httpx
    except ModuleNotFoundError:
        import httpx2 as httpx

    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "k")
    resp = httpx.Response(400, request=httpx.Request("POST", "https://x"))
    err = _openai.BadRequestError("This model doesn't support the temperature field.", response=resp, body=None)
    fake = fake_openai([err, _chat_ok()])
    run(BedrockLLM(model_cfg(name="openai.gpt-6.1-sol",
                             api_base="https://bedrock-mantle.us-east-1.api.aws/v1")).generate("x"))
    assert fake.calls[0]["temperature"] == 0.7
    assert "temperature" not in fake.calls[1]


@pytest.mark.parametrize("msg", [
    "model `openai.gpt-6.1-sol` isn't supported on this route",
    "The model 'anthropic.claude-sonnet-5' does not support the '/v1/chat/completions' API",
])
def test_openai_transport_route_unsupported_is_not_learned(fake_openai, monkeypatch, msg):
    import openai as _openai

    try:
        import httpx
    except ModuleNotFoundError:
        import httpx2 as httpx

    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "k")
    resp = httpx.Response(400, request=httpx.Request("POST", "https://x"))
    fake = fake_openai([_openai.BadRequestError(msg, response=resp, body=None)])
    llm = BedrockLLM(model_cfg(name="openai.gpt-6.1-sol", reasoning_effort="low",
                               api_base="https://bedrock-mantle.us-east-1.api.aws/v1"))
    with pytest.raises(ValueError, match="Programmatic Access"):
        run(llm.generate("x"))
    assert len(fake.calls) == 1 and not prov._quirks


@pytest.mark.parametrize("model,expected", [
    ("openai.gpt-oss-120b-1:0", "OK"),
    ("us.openai.gpt-6.1-sol", "<reasoning>Just reply OK.</reasoning>OK"),  # only gpt-oss
])
def test_openai_transport_strips_inline_gpt_oss_reasoning(fake_openai, monkeypatch, model, expected):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "k")
    fake_openai([_chat_ok(text="<reasoning>Just reply OK.</reasoning>OK")])
    llm = BedrockLLM(model_cfg(name=model,
                               api_base="https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1"))
    assert run(llm.generate("x")) == expected


def test_openai_transport_prices_cache_writes(fake_openai, monkeypatch):
    # GPT-6.1-sol on Bedrock, first call: prompt_tokens includes the 5411 written.
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "k")
    fake_openai([_chat_ok(pt=5413, ct=5, written=5411)])
    run(BedrockLLM(model_cfg(name="openai.gpt-6.1-sol",
                             api_base="https://bedrock-mantle.us-east-1.api.aws/openai/v1")).generate("x"))
    rec = read_ledger()[0]
    assert (rec["input_tokens"], rec["cache_write_tokens"], rec["cache_read_tokens"]) == (2, 5411, 0)


def test_openai_transport_switches_to_max_completion_tokens(fake_openai, monkeypatch):
    import openai as _openai

    try:
        import httpx
    except ModuleNotFoundError:
        import httpx2 as httpx

    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "k")
    resp = httpx.Response(400, request=httpx.Request("POST", "https://x"))
    err = _openai.BadRequestError(
        "Error code: 400 - Unsupported parameter: 'max_tokens' is not supported with this model.",
        response=resp, body=None)
    fake = fake_openai([err, _chat_ok(), _chat_ok()])
    llm = BedrockLLM(model_cfg(name="openai.gpt-6.1-sol",
                               api_base="https://bedrock-mantle.us-east-1.api.aws/openai/v1"))
    run(llm.generate("x"))
    run(llm.generate("y"))
    assert fake.calls[0]["max_tokens"] == 512
    assert [c.get("max_completion_tokens") for c in fake.calls[1:]] == [512, 512]
    assert not any("max_tokens" in c for c in fake.calls[1:])


def test_gpt_temperature_rejection_wording_is_learned():
    msg = ("Error code: 400 - Unsupported value: 'temperature' does not support 0.7 with this "
           "model. Only the default (1) value is supported.")
    assert prov._classify_rejection(msg, {"sampling", "max_tokens"}) == prov.NO_SAMPLING


def test_openai_transport_requires_key_or_credentials(monkeypatch):
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(prov, "_aws_credentials", lambda profile: None)
    with pytest.raises(ValueError, match="API key or AWS credentials"):
        BedrockLLM(model_cfg(api_base="https://bedrock-mantle.us-east-1.api.aws/v1"))


def _static_credentials(monkeypatch):
    from botocore.credentials import Credentials

    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-for-bedrock")
    monkeypatch.setattr(prov, "_aws_credentials", lambda profile: Credentials("AKIDEXAMPLE", "secret"))


def test_openai_transport_mints_key_from_aws_credentials(monkeypatch):
    import base64

    import openai as _openai

    _static_credentials(monkeypatch)
    keys = []
    fake = _FakeOpenAI([_chat_ok()])
    monkeypatch.setattr(_openai, "OpenAI", lambda **k: keys.append(k["api_key"]) or fake)
    llm = BedrockLLM(model_cfg(api_base="https://bedrock-mantle.us-east-1.api.aws/v1"))
    assert llm.transport.region == "us-east-1"
    run(llm.generate("x"))
    assert keys[0].startswith("bedrock-api-key-")  # preferred over OPENAI_API_KEY
    url = base64.b64decode(keys[0][len("bedrock-api-key-"):]).decode()
    assert url.startswith("bedrock.amazonaws.com/?Action=CallWithBearerToken")
    assert "us-east-1%2Fbedrock%2Faws4_request" in url and url.endswith("&Version=1")


def test_openai_transport_re_mints_aged_key(monkeypatch):
    import openai as _openai

    _static_credentials(monkeypatch)
    made = []
    monkeypatch.setattr(_openai, "OpenAI", lambda **k: made.append(k) or object())
    t = BedrockLLM(model_cfg(api_base="https://bedrock-mantle.us-west-2.api.aws/v1")).transport
    assert t.client is t.client and len(made) == 1
    t._minted_at -= prov._MINT_REFRESH_S + 1
    t.client
    assert len(made) == 2


def test_region_from_url():
    assert prov._region_from_url("https://bedrock-mantle.eu-west-1.api.aws/v1") == "eu-west-1"
    assert prov._region_from_url(
        "https://bedrock-runtime.ap-southeast-2.amazonaws.com/openai/v1") == "ap-southeast-2"
    assert prov._region_from_url("http://127.0.0.1:8080/v1") is None


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


def test_ensemble_mixes_transports_per_model(monkeypatch):
    """A model-level api_base overrides the llm-level one, in parent and worker."""
    from dataclasses import asdict

    from openevolve.config import Config, LLMModelConfig
    from openevolve.llm.ensemble import LLMEnsemble

    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "k")
    cfg = Config.from_dict({"llm": {"provider": "bedrock", "api_base": "aws://us-west-2", "models": [
        {"name": "us.anthropic.claude-haiku-5-5", "weight": 0.6},
        {"name": "us.openai.gpt-6.1-sol", "weight": 0.2,
         "api_base": "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1"},
        {"name": "openai.gpt-oss-120b", "weight": 0.2,
         "api_base": "https://bedrock-mantle.us-west-2.api.aws/v1"},
    ]}})
    for models in (cfg.llm.models, [LLMModelConfig(**asdict(m)) for m in cfg.llm.models]):
        ens = LLMEnsemble(models)
        assert [m.transport.name for m in ens.models] == ["converse", "openai", "openai"]
        assert ens.models[2].transport.base_url == "https://bedrock-mantle.us-west-2.api.aws/v1"


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
