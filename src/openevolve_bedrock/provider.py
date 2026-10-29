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
``bedrock`` LLM provider for OpenEvolve.

Two transports, chosen by ``api_base``:

* ``aws://<region>`` or unset  -> Converse API via boto3 (SigV4 credentials,
  IAM roles, or a Bedrock API key in ``AWS_BEARER_TOKEN_BEDROCK``). Works with
  any Bedrock text model from any vendor; supports prompt caching of the system
  message, cross-region inference profiles, Claude thinking and other vendors'
  reasoning effort.
* ``https://...``               -> OpenAI-compatible Chat Completions
  (``bedrock-mantle.<region>.api.aws/v1`` or
  ``bedrock-runtime.<region>.amazonaws.com/openai/v1``) with a Bedrock API key.
  Parameters Bedrock rejects (``seed``, ``top_p: null``) are never sent.

Models differ in what they accept: many reject the cache point, current ones
reject non-default ``temperature``, Claude thinking is ``adaptive`` on newer
models and ``budget_tokens`` on older ones, and vendors name reasoning effort
differently. Instead of a model table that goes stale, each transport sends the
current-generation shape and, when Bedrock rejects a field, drops or swaps it
for that model, retries, and remembers for the rest of the process (``_quirks``).

Every call appends one line to the usage ledger (see ``ledger.py``).

Environment knobs (all optional):

    OE_BEDROCK_REGION          default region for aws:// without one
    OE_BEDROCK_PROFILE         boto3 profile name
    OE_BEDROCK_CACHE           auto (default) | off   system-message cache point
    OE_BEDROCK_LEDGER          ledger path; "off" disables
    OE_BEDROCK_MAX_ATTEMPTS    throttle/transient retry budget (default 8)
    OE_BEDROCK_THINKING_BUDGET Claude thinking budget tokens; if set, enables
                               thinking regardless of reasoning_effort (models
                               that only take adaptive thinking get that)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import random
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from openevolve.llm.base import LLMInterface

from openevolve_bedrock import ledger as _ledger
from openevolve_bedrock.ledger import Usage

logger = logging.getLogger(__name__)

PROVIDER_NAME = "bedrock"

_THROTTLE_CODES = {
    "ThrottlingException",
    "TooManyRequestsException",
    "ServiceUnavailableException",
    "ModelNotReadyException",
    "InternalServerException",
    "ServiceQuotaExceededException",
}

# reasoning_effort -> Claude thinking budget (tokens), for models without adaptive thinking
_THINKING_BUDGETS = {"minimal": 1024, "low": 1024, "medium": 4096, "high": 16384,
                     "xhigh": 32768, "max": 32768}
# reasoning_effort -> Claude output_config.effort, for adaptive thinking
_EFFORTS = {"minimal": "low"}

# Cross-region inference profile prefixes
_PROFILE_PREFIXES = ("us.", "eu.", "apac.", "global.", "jp.", "au.", "ca.")


def _base_model_id(model: str) -> str:
    for p in _PROFILE_PREFIXES:
        if model.startswith(p):
            return model[len(p):]
    return model


def _vendor(model: str) -> str:
    """'anthropic' from 'us.anthropic.claude-...', 'openai' from 'openai.gpt-oss-...'."""
    return _base_model_id(model).split(".", 1)[0].lower()


class Settings:
    """Environment-derived settings, read once per process."""

    def __init__(self) -> None:
        self.region = os.environ.get("OE_BEDROCK_REGION") or os.environ.get("AWS_REGION") \
            or os.environ.get("AWS_DEFAULT_REGION")
        self.profile = os.environ.get("OE_BEDROCK_PROFILE") or None
        self.cache = (os.environ.get("OE_BEDROCK_CACHE") or "auto").strip().lower()
        try:
            self.max_attempts = max(1, int(os.environ.get("OE_BEDROCK_MAX_ATTEMPTS", "8")))
        except ValueError:
            self.max_attempts = 8
        tb = os.environ.get("OE_BEDROCK_THINKING_BUDGET")
        self.thinking_budget: Optional[int] = int(tb) if tb and tb.isdigit() else None


def _backoff(attempt: int, base: float = 1.0, cap: float = 30.0) -> float:
    return min(cap, base * (2 ** (attempt - 1))) * random.uniform(0.5, 1.5)


# --------------------------------------------------------------------------
# Transports
# --------------------------------------------------------------------------


class _Transport:
    name = "base"

    def complete(
        self, system: Optional[str], messages: List[Dict[str, str]], params: Dict[str, Any]
    ) -> Tuple[str, Usage]:
        raise NotImplementedError


_client_cache: Dict[Tuple[str, Optional[str], int], Any] = {}
_client_lock = threading.Lock()

# Request features a model rejected, learned from Bedrock's errors (per process).
NO_CACHE = "no_cache"                    # cachePoint
NO_SAMPLING = "no_sampling"              # temperature / topP
THINKING_BUDGET = "thinking_budget"      # adaptive rejected -> budget_tokens
THINKING_ADAPTIVE = "thinking_adaptive"  # budget_tokens rejected -> adaptive
NO_THINKING = "no_thinking"              # neither accepted
REASONING_NESTED = "reasoning_nested"    # reasoning_effort rejected -> reasoning.effort
NO_REASONING = "no_reasoning"            # neither accepted
_quirks: Dict[str, set] = {}
_quirks_lock = threading.Lock()


def _has_quirk(model: str, quirk: str) -> bool:
    with _quirks_lock:
        return quirk in _quirks.get(model, ())


def _learn_quirk(model: str, quirk: str, message: str) -> bool:
    """Record a quirk; False if already known (the caller should then give up)."""
    with _quirks_lock:
        known = _quirks.setdefault(model, set())
        if quirk in known:
            return False
        known.add(quirk)
    logger.warning("%s: %s; adapting requests to this model (%s)", model, quirk, message[:200])
    return True


def _classify_rejection(message: str, features: set) -> Optional[str]:
    """Map a Bedrock rejection to the quirk that avoids it, given the features sent."""
    m = message.lower()
    if "cache" in features and "cach" in m:
        return NO_CACHE
    if "sampling" in features and ("temperature" in m or "top_p" in m or "topp" in m):
        return NO_SAMPLING
    if "thinking_adaptive" in features and "adaptive" in m:
        return THINKING_BUDGET
    if "thinking_budget" in features and ("thinking.type.enabled" in m or "budget_tokens" in m):
        return THINKING_ADAPTIVE
    if "reasoning_effort" in features and "reasoning_effort" in m:
        return REASONING_NESTED
    if "reasoning_nested" in features and "reasoning" in m:
        return NO_REASONING
    return None


def _learn_from_rejection(model: str, message: str, features: set) -> bool:
    quirk = _classify_rejection(message, features)
    if quirk is None and features & {"reasoning_effort", "reasoning_nested"}:
        # Some models reject reasoning fields without naming them (DeepSeek R1:
        # "Validation Error"). Reasoning effort is optional, so it goes first.
        quirk = NO_REASONING
    if quirk is None:
        return False
    # Thinking and reasoning fall back one shape, then off.
    if quirk == THINKING_BUDGET and _has_quirk(model, THINKING_ADAPTIVE):
        quirk = NO_THINKING
    elif quirk == THINKING_ADAPTIVE and _has_quirk(model, THINKING_BUDGET):
        quirk = NO_THINKING
    return _learn_quirk(model, quirk, message)


def _thinking_fields(model: str, params: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Claude thinking request fields for this model, and which shape was chosen."""
    effort, budget = params.get("effort"), params.get("thinking_budget")
    if not (effort or budget) or _has_quirk(model, NO_THINKING):
        return None, None
    use_budget = (budget and not _has_quirk(model, THINKING_ADAPTIVE)) or _has_quirk(model, THINKING_BUDGET)
    if use_budget:
        budget = int(budget or _THINKING_BUDGETS.get(str(effort).lower(), 4096))
        return {"thinking": {"type": "enabled", "budget_tokens": budget}}, "thinking_budget"
    fields: Dict[str, Any] = {"thinking": {"type": "adaptive"}}
    if effort:
        fields["output_config"] = {"effort": _EFFORTS.get(str(effort).lower(), str(effort).lower())}
    return fields, "thinking_adaptive"


def _reasoning_fields(model: str, effort: Optional[str]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Non-Anthropic reasoning effort: ``reasoning_effort``, else ``reasoning.effort``."""
    if not effort or _has_quirk(model, NO_REASONING):
        return None, None
    if _has_quirk(model, REASONING_NESTED):
        return {"reasoning": {"effort": str(effort)}}, "reasoning_nested"
    return {"reasoning_effort": str(effort)}, "reasoning_effort"


def _bedrock_runtime_client(region: str, profile: Optional[str], timeout: int):
    key = (region, profile, timeout)
    with _client_lock:
        if key in _client_cache:
            return _client_cache[key]
        import boto3
        from botocore.config import Config

        session = boto3.Session(profile_name=profile, region_name=region)
        client = session.client(
            "bedrock-runtime",
            config=Config(
                read_timeout=timeout,
                connect_timeout=10,
                # We do our own throttle-aware retries so the ledger can count them.
                retries={"total_max_attempts": 1},
            ),
        )
        _client_cache[key] = client
        return client


class ConverseTransport(_Transport):
    name = "converse"

    def __init__(self, model: str, region: str, settings: Settings, timeout: int):
        self.model = model
        self.region = region
        self.settings = settings
        self.timeout = timeout
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = _bedrock_runtime_client(self.region, self.settings.profile, self.timeout)
        return self._client

    def _cache_enabled(self) -> bool:
        return self.settings.cache != "off" and not _has_quirk(self.model, NO_CACHE)

    def build_request(
        self,
        system: Optional[str],
        messages: List[Dict[str, str]],
        params: Dict[str, Any],
        use_cache: bool,
    ) -> Tuple[Dict[str, Any], set]:
        """The Converse request, and the optional features it uses (for rejection handling)."""
        features: set = set()
        req: Dict[str, Any] = {
            "modelId": self.model,
            "messages": [
                {"role": m["role"], "content": [{"text": m["content"]}]}
                for m in messages
                if m.get("role") in ("user", "assistant")
            ],
        }
        if system:
            sys_blocks: List[Dict[str, Any]] = [{"text": system}]
            if use_cache:
                sys_blocks.append({"cachePoint": {"type": "default"}})
                features.add("cache")
            req["system"] = sys_blocks

        extra: Dict[str, Any] = {}
        if params.get("vendor") == "anthropic":
            fields, shape = _thinking_fields(self.model, params)
        else:
            fields, shape = _reasoning_fields(self.model, params.get("effort"))
        if fields:
            extra.update(fields)
            features.add(shape)

        inference: Dict[str, Any] = {}
        max_tokens = params.get("max_tokens")
        if max_tokens is not None:
            inference["maxTokens"] = int(max_tokens)
        if shape == "thinking_budget":
            # maxTokens must exceed the budget
            budget = extra["thinking"]["budget_tokens"]
            inference["maxTokens"] = max(int(max_tokens or 0), budget + 1024)
        # Claude thinking forbids sampling params; some models reject them outright.
        if not (shape or "").startswith("thinking") and not _has_quirk(self.model, NO_SAMPLING):
            if params.get("temperature") is not None:
                inference["temperature"] = float(params["temperature"])
            if params.get("top_p") is not None:
                inference["topP"] = float(params["top_p"])
            if "temperature" in inference or "topP" in inference:
                features.add("sampling")
        if inference:
            req["inferenceConfig"] = inference
        if extra:
            req["additionalModelRequestFields"] = extra
        return req, features

    def complete(self, system, messages, params):
        from botocore.exceptions import ClientError, EndpointConnectionError, ReadTimeoutError

        attempt = 0
        throttled = 0
        while True:
            attempt += 1
            use_cache = bool(system) and self._cache_enabled()
            req, features = self.build_request(system, messages, params, use_cache)
            t0 = time.monotonic()
            try:
                resp = self.client.converse(**req)
                break
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "")
                msg = str(e.response.get("Error", {}).get("Message", e))
                # Unsupported fields come back as ValidationException; an unsupported
                # cachePoint as AccessDeniedException ("did not allow prompt caching").
                if code in ("ValidationException", "AccessDeniedException") and \
                        _learn_from_rejection(self.model, msg, features):
                    attempt -= 1  # does not count against the budget
                    continue
                if code in _THROTTLE_CODES and attempt < self.settings.max_attempts:
                    throttled += 1
                    delay = _backoff(attempt)
                    logger.warning("%s on %s (attempt %d/%d); sleeping %.1fs",
                                   code, self.model, attempt, self.settings.max_attempts, delay)
                    time.sleep(delay)
                    continue
                raise
            except (EndpointConnectionError, ReadTimeoutError):
                if attempt < self.settings.max_attempts:
                    delay = _backoff(attempt)
                    logger.warning("connection/timeout on %s (attempt %d); sleeping %.1fs",
                                   self.model, attempt, delay)
                    time.sleep(delay)
                    continue
                raise
        latency_ms = (time.monotonic() - t0) * 1000.0

        blocks = (resp.get("output", {}).get("message", {}) or {}).get("content", []) or []
        text = "".join(b.get("text", "") for b in blocks if isinstance(b, dict) and "text" in b)
        u = resp.get("usage", {}) or {}
        usage = Usage(
            model=self.model,
            transport=self.name,
            input_tokens=int(u.get("inputTokens", 0) or 0),
            output_tokens=int(u.get("outputTokens", 0) or 0),
            cache_read_tokens=int(u.get("cacheReadInputTokens", 0) or 0),
            cache_write_tokens=int(u.get("cacheWriteInputTokens", 0) or 0),
            latency_ms=latency_ms,
            attempts=attempt,
            throttled=throttled,
            stop_reason=resp.get("stopReason"),
        )
        return text, usage


class OpenAICompatTransport(_Transport):
    """Bedrock's OpenAI-compatible Chat Completions (Mantle or runtime /openai/v1)."""

    name = "openai"

    def __init__(self, model: str, base_url: str, api_key: Optional[str], settings: Settings,
                 timeout: int):
        self.model = model
        self.base_url = base_url
        self.settings = settings
        self.timeout = timeout
        self.api_key = (
            api_key
            or os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
            or os.environ.get("OPENAI_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "openevolve-bedrock: OpenAI-compatible transport needs a Bedrock API key. "
                "Set llm.api_key in config (e.g. \"${AWS_BEARER_TOKEN_BEDROCK}\") or export "
                "AWS_BEARER_TOKEN_BEDROCK."
            )
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import openai

            self._client = openai.OpenAI(
                api_key=self.api_key, base_url=self.base_url, timeout=self.timeout, max_retries=0
            )
        return self._client

    def build_request(self, system, messages, params) -> Tuple[Dict[str, Any], set]:
        """The Chat Completions request, and the optional features it uses."""
        features: set = set()
        msgs: List[Dict[str, str]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend({"role": m["role"], "content": m["content"]} for m in messages)
        req: Dict[str, Any] = {"model": self.model, "messages": msgs, "stream": False}
        if params.get("max_tokens") is not None:
            req["max_tokens"] = int(params["max_tokens"])
        if not _has_quirk(self.model, NO_SAMPLING):
            if params.get("temperature") is not None:
                req["temperature"] = float(params["temperature"])
            if params.get("top_p") is not None:  # never send top_p: null (openevolve #414)
                req["top_p"] = float(params["top_p"])
            if "temperature" in req or "top_p" in req:
                features.add("sampling")
        fields, shape = _reasoning_fields(self.model, params.get("effort"))
        if fields:
            req.update(fields)
            features.add(shape)
        # Deliberately no `seed`: Bedrock's OpenAI-compatible endpoints do not accept it.
        return req, features

    def complete(self, system, messages, params):
        import openai

        attempt = 0
        throttled = 0
        while True:
            attempt += 1
            req, features = self.build_request(system, messages, params)
            t0 = time.monotonic()
            try:
                resp = self.client.chat.completions.create(**req)
                break
            except openai.BadRequestError as e:
                if _learn_from_rejection(self.model, str(e), features):
                    attempt -= 1  # does not count against the budget
                    continue
                raise
            except openai.RateLimitError:
                if attempt < self.settings.max_attempts:
                    throttled += 1
                    delay = _backoff(attempt)
                    logger.warning("429 on %s (attempt %d); sleeping %.1fs", self.model, attempt, delay)
                    time.sleep(delay)
                    continue
                raise
            except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError):
                if attempt < self.settings.max_attempts:
                    delay = _backoff(attempt)
                    logger.warning("transient error on %s (attempt %d); sleeping %.1fs",
                                   self.model, attempt, delay)
                    time.sleep(delay)
                    continue
                raise
        latency_ms = (time.monotonic() - t0) * 1000.0
        if isinstance(resp, str):
            raise ValueError("endpoint returned a raw string instead of a chat completion")
        choice = resp.choices[0]
        text = choice.message.content or ""
        u = getattr(resp, "usage", None)
        prompt_tokens = int(getattr(u, "prompt_tokens", 0) or 0) if u else 0
        completion_tokens = int(getattr(u, "completion_tokens", 0) or 0) if u else 0
        cached = 0
        details = getattr(u, "prompt_tokens_details", None) if u else None
        if details is not None:
            cached = int(getattr(details, "cached_tokens", 0) or 0)
        usage = Usage(
            model=self.model,
            transport=self.name,
            # prompt_tokens includes cached tokens; the ledger prices them separately
            input_tokens=max(0, prompt_tokens - cached),
            output_tokens=completion_tokens,
            cache_read_tokens=cached,
            latency_ms=latency_ms,
            attempts=attempt,
            throttled=throttled,
            stop_reason=getattr(choice, "finish_reason", None),
        )
        return text, usage


# --------------------------------------------------------------------------
# The OpenEvolve-facing class
# --------------------------------------------------------------------------


class BedrockLLM(LLMInterface):
    """OpenEvolve ``LLMInterface`` backed by Amazon Bedrock."""

    def __init__(self, model_cfg) -> None:
        self.model: str = model_cfg.name
        if not self.model:
            raise ValueError("openevolve-bedrock: model `name` is required")
        self.system_message = getattr(model_cfg, "system_message", None)
        self.temperature = getattr(model_cfg, "temperature", None)
        self.top_p = getattr(model_cfg, "top_p", None)
        self.max_tokens = getattr(model_cfg, "max_tokens", None) or 4096
        self.timeout = int(getattr(model_cfg, "timeout", None) or 120)
        self.api_base = getattr(model_cfg, "api_base", None)
        self.api_key = getattr(model_cfg, "api_key", None)
        self.reasoning_effort = getattr(model_cfg, "reasoning_effort", None)
        self.last_usage: Optional[Dict[str, Any]] = None
        self.settings = Settings()
        self.transport = self._make_transport()
        self._seed_warned = False

        logger.info("Initialized Bedrock LLM: model=%s transport=%s", self.model, self.transport.name)

    # -- construction -----------------------------------------------------

    def _make_transport(self) -> _Transport:
        base = (self.api_base or "").strip()
        if base.startswith("http://") or base.startswith("https://"):
            return OpenAICompatTransport(self.model, base, self.api_key, self.settings, self.timeout)
        region = None
        if base.startswith("aws://"):
            region = base[len("aws://"):].strip("/") or None
        elif base and base != "https://api.openai.com/v1":
            # Anything else we don't understand: treat as a region name.
            region = base
        region = region or self.settings.region
        if not region:
            raise ValueError(
                "openevolve-bedrock: no AWS region. Set llm.api_base to aws://<region>, or "
                "export OE_BEDROCK_REGION / AWS_REGION."
            )
        return ConverseTransport(self.model, region, self.settings, self.timeout)

    # -- parameter mapping -----------------------------------------------

    def _params(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        if kwargs.get("seed", None) is not None and not self._seed_warned:
            logger.debug("Bedrock does not accept `seed`; ignoring it for %s", self.model)
            self._seed_warned = True

        # The transports turn these into each model's thinking/reasoning fields.
        params["vendor"] = _vendor(self.model)
        params["effort"] = kwargs.get("reasoning_effort", self.reasoning_effort)
        if self.settings.thinking_budget and params["vendor"] == "anthropic":
            params["thinking_budget"] = self.settings.thinking_budget
        return params

    # -- LLMInterface ----------------------------------------------------

    async def generate(self, prompt: str, **kwargs) -> str:
        return await self.generate_with_context(
            system_message=self.system_message,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        params = self._params(kwargs)
        loop = asyncio.get_event_loop()
        text, usage = await loop.run_in_executor(
            None, lambda: self.transport.complete(system_message, messages, params)
        )
        self.last_usage = usage.as_openevolve_usage()
        try:
            _ledger.append(usage)
        except OSError as e:  # never let bookkeeping kill the run
            logger.warning("ledger write failed: %s", e)
        logger.info(
            "LLM Token Usage (%s): prompt_tokens=%d, completion_tokens=%d, cache_read=%d, "
            "cache_write=%d, latency_ms=%.0f, throttled=%d",
            self.model, usage.input_tokens, usage.output_tokens, usage.cache_read_tokens,
            usage.cache_write_tokens, usage.latency_ms, usage.throttled,
        )
        return text


# --------------------------------------------------------------------------
# Registration with OpenEvolve
# --------------------------------------------------------------------------

_registered = False
_original_run_iteration_worker = None


def _iteration_seed(llm_seed: int, iteration: int) -> int:
    """Sampler seed for one iteration; mirrors OpenEvolve's own llm_seed derivation."""
    digest = hashlib.md5(f"{llm_seed}:{iteration}".encode(), usedforsecurity=False)
    return int(digest.hexdigest()[:8], 16)


def _seeded_run_iteration_worker(iteration: int, *args, **kwargs):
    """
    OpenEvolve's ``_run_iteration_worker`` with the ensemble sampler reseeded
    from ``(llm_seed, iteration)`` first.

    Upstream seeds every worker's ``LLMEnsemble`` with the same ``llm_seed``, so
    all workers draw the same model sequence (openevolve#554). Seeding per
    iteration makes draws independent across iterations and still reproducible,
    whichever worker runs which iteration. With ``random_seed: null`` the
    sampler is unseeded and already independent, so it is left alone.

    Pickled by reference to this module, so unpickling it in a spawned worker
    imports the package and registers the provider there.
    """
    from openevolve import process_parallel as pp

    try:
        pp._lazy_init_worker_components()
        models = pp._worker_config.llm.models
        llm_seed = getattr(models[0], "random_seed", None) if models else None
        if llm_seed is not None:
            pp._worker_llm_ensemble.random_state.seed(_iteration_seed(llm_seed, iteration))
    except Exception:  # leave error reporting to the original
        logger.debug("per-iteration reseed skipped", exc_info=True)
    return _original_run_iteration_worker(iteration, *args, **kwargs)


def _patch_iteration_worker() -> None:
    global _original_run_iteration_worker
    try:
        from openevolve import process_parallel as pp
    except ImportError:
        return
    current = getattr(pp, "_run_iteration_worker", None)
    if current is None or not hasattr(pp, "_lazy_init_worker_components"):
        logger.debug("OpenEvolve layout not recognised; per-iteration seeding not installed")
        return
    if current is not _seeded_run_iteration_worker:
        _original_run_iteration_worker = current
        pp._run_iteration_worker = _seeded_run_iteration_worker


def register() -> None:
    """Register ``provider: bedrock`` with OpenEvolve's LLM ensemble (idempotent)."""
    global _registered
    if _registered:
        return
    from openevolve.llm import ensemble

    registry = getattr(ensemble, "_PROVIDER_REGISTRY", None)
    if isinstance(registry, dict):
        registry[PROVIDER_NAME] = lambda cfg: BedrockLLM(cfg)
    else:  # older OpenEvolve without a registry: wrap the factory
        original = ensemble._create_model

        def _create_model(model_cfg):
            if getattr(model_cfg, "provider", None) == PROVIDER_NAME and not getattr(
                model_cfg, "init_client", None
            ):
                return BedrockLLM(model_cfg)
            return original(model_cfg)

        ensemble._create_model = _create_model
    _patch_iteration_worker()
    _registered = True
