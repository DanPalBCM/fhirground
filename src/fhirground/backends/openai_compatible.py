"""OpenAI-compatible backend, for vLLM and anything speaking the same API.

**Request shape verified** against the OpenAI client's own type definitions
(``openai.types.shared_params.response_format_json_schema``) rather than from
memory: ``response_format = {"type": "json_schema", "json_schema": {"name",
"schema", "strict"}}``. See ``docs/decisions.md`` D-003.

Two constraint mechanisms, because servers differ:

* ``response_format: json_schema`` -- the OpenAI-compatible spelling, which
  recent vLLM supports and which gives true token-level constrained decoding
  when the server's guided backend is active.
* ``guided_json`` -- vLLM's older extra-body parameter, kept as a fallback for
  servers that predate the former.

``strict`` defaults to **False**. OpenAI's strict mode requires every property
to appear in ``required``, and the mention schema deliberately has four
required fields and fourteen optional ones -- a model forced to emit every
field invents values for the ones it cannot support. vLLM's guided decoding
constrains to the schema without that requirement, so strict buys nothing here
and costs correctness.

No test in this package calls a server; the request builder is tested with
``respx``.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from fhirground.backends.base import RawResponse
from fhirground.exceptions import BackendError

#: How the schema is attached to the request.
CONSTRAINT_MODES = ("json_schema", "guided_json", "none")


class OpenAICompatible:
    """Chat-completions backend for an OpenAI-compatible server.

    Args:
        base_url: Server base, e.g. ``http://localhost:8000/v1``.
        model: Model id the server exposes.
        api_key: Bearer token. vLLM usually needs none.
        temperature: Default sampling temperature.
        max_tokens: Completion cap. Too low truncates the JSON and burns a
            repair attempt on something the model cannot fix, so the default is
            generous.
        constraint_mode: ``json_schema`` (default), ``guided_json``, or
            ``none`` for a server with no constrained decoding at all.
        strict: Value for ``json_schema.strict``. Off by default; see the
            module docstring.
        timeout: Per-request timeout in seconds.
        client: Inject an ``httpx.Client`` for testing.
        extra_body: Additional request fields, merged last.
    """

    name = "openai_compatible"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        constraint_mode: str = "json_schema",
        strict: bool = False,
        timeout: float = 120.0,
        client: httpx.Client | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        if constraint_mode not in CONSTRAINT_MODES:
            raise BackendError(
                f"unknown constraint_mode {constraint_mode!r}; use one of {list(CONSTRAINT_MODES)}"
            )
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.model_id = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.constraint_mode = constraint_mode
        self.strict = strict
        self.timeout = timeout
        self.extra_body = dict(extra_body or {})
        self._client = client
        self._owns_client = client is None

    # -- plumbing -------------------------------------------------------

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def close(self) -> None:
        """Close the HTTP client if this backend created it."""
        if self._client is not None and self._owns_client:
            self._client.close()
            self._client = None

    def __enter__(self) -> OpenAICompatible:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    # -- the request builder (unit-tested directly) ---------------------

    def build_request(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        temperature: float | None = None,
        n: int = 1,
        system: str | None = None,
    ) -> dict[str, Any]:
        """Build the chat-completions request body.

        Separated from the call so SPEC §0's requirement -- a test that the
        mention schema is accepted by each backend's request builder -- can be
        met without a server.
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens,
        }
        if n > 1:
            body["n"] = n

        if self.constraint_mode == "json_schema":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.get("title") or "MentionList",
                    "schema": schema,
                    "strict": self.strict,
                },
            }
        elif self.constraint_mode == "guided_json":
            # vLLM's extra-body form, for servers predating response_format.
            body["guided_json"] = schema
            body["guided_decoding_backend"] = self.extra_body.pop(
                "guided_decoding_backend", "outlines"
            )

        body.update(self.extra_body)
        return body

    # -- the Backend protocol -------------------------------------------

    def extract(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        temperature: float = 0.0,
        n: int = 1,
        system: str | None = None,
    ) -> list[RawResponse]:
        """Call the server and return one response per sample."""
        body = self.build_request(prompt, schema, temperature=temperature, n=n, system=system)
        url = f"{self.base_url}/chat/completions"

        started = time.perf_counter()
        try:
            response = self._get_client().post(url, json=body, headers=self._headers())
        except httpx.HTTPError as exc:
            raise BackendError(
                f"request to {url} failed: {type(exc).__name__}. Is the server running?"
            ) from exc
        elapsed = time.perf_counter() - started

        if response.status_code == 401:
            raise BackendError(f"{url} rejected the credentials (401)")
        if response.status_code == 404:
            raise BackendError(
                f"{url} returned 404. Check --base-url includes the API prefix, usually '/v1'."
            )
        if response.status_code >= 400:
            raise BackendError(f"{url} returned {response.status_code}: {response.text[:300]}")

        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise BackendError(f"{url} returned a non-JSON body: {exc.msg}") from exc

        choices = payload.get("choices") or []
        if not choices:
            raise BackendError(f"{url} returned no choices")

        usage = payload.get("usage") or {}
        model_id = str(payload.get("model") or self.model)
        out: list[RawResponse] = []
        for choice in choices:
            message = choice.get("message") or {}
            out.append(
                RawResponse(
                    text=str(message.get("content") or ""),
                    model_id=model_id,
                    tokens_in=usage.get("prompt_tokens"),
                    tokens_out=usage.get("completion_tokens"),
                    latency_s=round(elapsed, 4),
                    logprob=_mean_logprob(choice),
                    finish_reason=choice.get("finish_reason"),
                    raw=choice,
                )
            )
        return out

    def describe(self) -> dict[str, Any]:
        """Configuration summary."""
        return {
            "backend": self.name,
            "base_url": self.base_url,
            "model_id": self.model_id,
            "constraint_mode": self.constraint_mode,
            "strict": self.strict,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


def _mean_logprob(choice: dict[str, Any]) -> float | None:
    """Mean token probability from a choice, when logprobs were returned.

    Converted from log space to a probability so the confidence component is
    on the same [0, 1] scale as the others.
    """
    import math

    logprobs = choice.get("logprobs") or {}
    tokens = logprobs.get("content") or []
    values: list[float] = [
        float(t["logprob"]) for t in tokens if isinstance(t, dict) and t.get("logprob") is not None
    ]
    if not values:
        return None
    try:
        return round(math.exp(sum(values) / len(values)), 6)
    except (OverflowError, ValueError):  # pragma: no cover
        return None
