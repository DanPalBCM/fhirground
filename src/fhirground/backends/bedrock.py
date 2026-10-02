"""Bedrock backend, using the Converse API with a single forced tool.

**Request shape verified** against botocore's own service model for
``bedrock-runtime`` rather than from documentation or memory
(``docs/decisions.md`` D-003):

* ``Converse`` accepts ``toolConfig`` with ``tools`` (required) and
  ``toolChoice``;
* ``toolSpec`` requires ``name`` and ``inputSchema``, and also offers
  ``description`` and ``strict``;
* ``inputSchema`` is a union whose only member is ``json``, a free-form
  document -- so the mention schema goes in as-is;
* ``toolChoice`` supports ``any``, ``auto``, and ``tool: {name}``.

Converse has no ``response_format``-style structured output, so the forced-tool
approach is the right one: one tool, ``toolChoice.tool``, and the model's
arguments *are* the mention JSON.

``boto3`` is an optional extra. Credentials come from the standard boto3 chain,
never from fhirground configuration -- this package should not be in the
business of holding AWS keys.
"""

from __future__ import annotations

import json
import time
from typing import Any

from fhirground.backends.base import RawResponse
from fhirground.exceptions import BackendError

#: Name of the single forced tool. The model sees this, so it reads as an
#: instruction rather than an identifier.
TOOL_NAME = "record_clinical_mentions"

TOOL_DESCRIPTION = (
    "Record every clinical finding extracted from the note. Call this exactly once "
    "with all findings. Do not include medical codes."
)

#: Bedrock error codes that mean "the service said no", not "the model was wrong".
_AUTH_ERRORS = frozenset(
    {
        "AccessDeniedException",
        "UnrecognizedClientException",
        "InvalidSignatureException",
        "ExpiredTokenException",
    }
)
_THROTTLE_ERRORS = frozenset({"ThrottlingException", "TooManyRequestsException"})


class BedrockBackend:
    """Converse-API backend with a forced tool carrying the mention schema.

    Args:
        model_id: Bedrock model id or inference profile ARN.
        region: AWS region.
        temperature: Default sampling temperature.
        max_tokens: Completion cap.
        client: Inject a ``bedrock-runtime`` client for testing. Tests use
            botocore's ``Stubber``, so no test here reaches AWS.
        strict: Value for ``toolSpec.strict``, when the service supports it.
            Off by default for the same reason as the OpenAI backend: the
            mention schema has mostly-optional fields on purpose.
        additional_model_request_fields: Passed through to Converse for
            model-specific options.
    """

    name = "bedrock"

    def __init__(
        self,
        *,
        model_id: str,
        region: str = "us-east-1",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        client: Any = None,
        strict: bool = False,
        additional_model_request_fields: dict[str, Any] | None = None,
    ) -> None:
        self.model_id = model_id
        self.region = region
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.strict = strict
        self.additional_model_request_fields = dict(additional_model_request_fields or {})
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import boto3
            except ImportError as exc:
                raise BackendError(
                    "the Bedrock backend needs the optional extra: "
                    "pip install 'fhirground[bedrock]'"
                ) from exc
            self._client = boto3.client("bedrock-runtime", region_name=self.region)
        return self._client

    # -- the request builder (unit-tested with Stubber) -----------------

    def build_request(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        temperature: float | None = None,
        system: str | None = None,
    ) -> dict[str, Any]:
        """Build the ``Converse`` request keyword arguments.

        The mention schema goes into ``toolConfig.tools[0].toolSpec.inputSchema.json``
        unchanged, and ``toolChoice`` forces that one tool so the model cannot
        answer in prose.
        """
        tool_spec: dict[str, Any] = {
            "name": TOOL_NAME,
            "description": TOOL_DESCRIPTION,
            "inputSchema": {"json": schema},
        }
        if self.strict:
            tool_spec["strict"] = True

        request: dict[str, Any] = {
            "modelId": self.model_id,
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {
                "temperature": self.temperature if temperature is None else temperature,
                "maxTokens": self.max_tokens,
            },
            "toolConfig": {
                "tools": [{"toolSpec": tool_spec}],
                "toolChoice": {"tool": {"name": TOOL_NAME}},
            },
        }
        if system:
            request["system"] = [{"text": system}]
        if self.additional_model_request_fields:
            request["additionalModelRequestFields"] = self.additional_model_request_fields
        return request

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
        """Call Converse ``n`` times and return one response per call.

        Converse returns a single message per call, so ``n > 1`` means ``n``
        calls. That is the honest implementation: faking samples from one
        response would make the self-consistency component meaningless.
        """
        client = self._get_client()
        request = self.build_request(prompt, schema, temperature=temperature, system=system)

        out: list[RawResponse] = []
        for _ in range(max(1, n)):
            started = time.perf_counter()
            try:
                response = client.converse(**request)
            except Exception as exc:
                raise self._translate(exc) from exc
            elapsed = time.perf_counter() - started
            out.append(self._parse(response, elapsed))
        return out

    def _translate(self, exc: Exception) -> BackendError:
        """Turn a botocore error into a BackendError with a usable message."""
        code = ""
        response = getattr(exc, "response", None)
        if isinstance(response, dict):
            code = str((response.get("Error") or {}).get("Code") or "")

        if code in _AUTH_ERRORS:
            return BackendError(
                f"Bedrock rejected the credentials ({code}). Credentials come from the "
                "standard boto3 chain; check your profile, role, and region "
                f"({self.region})."
            )
        if code in _THROTTLE_ERRORS:
            return BackendError(
                f"Bedrock throttled the request ({code}). Lower --concurrency or request "
                "a quota increase."
            )
        if code == "ValidationException":
            return BackendError(
                f"Bedrock rejected the request ({code}): {exc}. If this names the tool "
                "schema, the mention schema has left the subset Bedrock accepts -- "
                "fhirground.schemas.assert_backend_safe checks for that."
            )
        if code == "ResourceNotFoundException":
            return BackendError(
                f"Bedrock has no model {self.model_id!r} in {self.region} ({code}). "
                "Check the model id and that access is enabled for your account."
            )
        return BackendError(f"Bedrock call failed: {type(exc).__name__}: {exc}")

    def _parse(self, response: dict[str, Any], elapsed: float) -> RawResponse:
        """Pull the tool input back out of a Converse response."""
        usage = response.get("usage") or {}
        output = response.get("output") or {}
        message = output.get("message") or {}
        blocks = message.get("content") or []

        tool_input: Any = None
        text_parts: list[str] = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if "toolUse" in block:
                use = block["toolUse"] or {}
                if use.get("name") == TOOL_NAME:
                    tool_input = use.get("input")
            elif "text" in block:
                text_parts.append(str(block["text"]))

        if tool_input is not None:
            text = json.dumps(tool_input)
        elif text_parts:
            # The model answered in prose despite toolChoice. Pass it through so
            # the repair loop can tell it what was wrong, rather than raising a
            # backend error for something the model can fix.
            text = "\n".join(text_parts)
        else:
            text = ""

        return RawResponse(
            text=text,
            model_id=self.model_id,
            tokens_in=usage.get("inputTokens"),
            tokens_out=usage.get("outputTokens"),
            latency_s=round(elapsed, 4),
            finish_reason=response.get("stopReason"),
            raw={"stopReason": response.get("stopReason"), "usage": usage},
        )

    def describe(self) -> dict[str, Any]:
        """Configuration summary."""
        return {
            "backend": self.name,
            "model_id": self.model_id,
            "region": self.region,
            "tool_name": TOOL_NAME,
            "strict": self.strict,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
