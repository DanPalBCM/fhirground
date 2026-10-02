"""The backend protocol.

A backend turns a prompt plus the mention schema into raw text. It does **not**
parse, validate, repair, or know about FHIR -- :mod:`fhirground.extract` does
that. Keeping the boundary there means a new backend is one small class, and
the repair loop is tested once rather than per backend.

Backends must not log note text. ``RawResponse`` carries the text because the
repair loop needs it, but every log line in this package records ids, counts,
timings, and hashes only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class RawResponse:
    """One backend response, before parsing.

    Attributes:
        text: The raw response body, expected to be JSON matching the mention
            schema.
        model_id: Which model produced it.
        tokens_in: Prompt tokens, when the backend reports them.
        tokens_out: Completion tokens, when reported.
        latency_s: Wall-clock seconds for the call.
        logprob: Mean token probability, when the backend exposes it. Feeds
            the ``l`` confidence component, which is weighted 0 by default
            because few backends report it comparably.
        finish_reason: Backend-reported stop reason. ``length`` means the
            response was truncated, which is worth surfacing: truncated JSON
            fails to parse and consumes a repair attempt for a reason the
            model cannot fix.
        cached: Whether this came from the on-disk cache rather than a call.
        raw: The untouched provider payload, for debugging.
    """

    text: str
    model_id: str = ""
    tokens_in: int | None = None
    tokens_out: int | None = None
    latency_s: float | None = None
    logprob: float | None = None
    finish_reason: str | None = None
    cached: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def truncated(self) -> bool:
        """Whether the backend said it ran out of room."""
        return (self.finish_reason or "").lower() in ("length", "max_tokens")


@runtime_checkable
class Backend(Protocol):
    """Anything that can produce mention JSON from a prompt."""

    name: str
    model_id: str

    def extract(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        temperature: float = 0.0,
        n: int = 1,
    ) -> list[RawResponse]:
        """Produce ``n`` responses for ``prompt``, constrained to ``schema``.

        Args:
            prompt: The rendered prompt, including the chunk text.
            schema: The mention JSON Schema, as the constrained-decoding
                target.
            temperature: Sampling temperature. ``n > 1`` with temperature 0
                yields identical samples, so self-consistency needs a non-zero
                value to mean anything.
            n: Number of samples.

        Returns:
            One :class:`RawResponse` per sample.

        Raises:
            BackendError: On a transport, auth, or throttling failure. This is
                distinct from a schema failure, so it does not consume a
                repair attempt -- the model was never at fault.
        """
        ...

    def describe(self) -> dict[str, Any]:
        """Configuration summary for the run summary and Provenance."""
        ...
