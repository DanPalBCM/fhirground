"""Chunk to mentions: the backend call, parsing, and the repair loop.

Three responsibilities, deliberately kept out of the backends:

1. **Parse** the raw text as JSON and validate it against the mention model.
   Models wrap JSON in prose and markdown fences; this strips what can be
   stripped without guessing at content.
2. **Repair** on failure, by re-prompting with the validation error. Capped at
   ``max_repairs`` (default 2), and the count feeds the ``v`` confidence
   component -- a mention that took two attempts is worth less than one that
   parsed first time.
3. **Cache** calls on disk, keyed by backend config, prompt, and chunk.

A transport or auth failure is **not** a repair case: the model was never at
fault, so :class:`~fhirground.exceptions.BackendError` propagates immediately
rather than burning attempts.

The cache contains **note text**, because the prompt does. That is documented
in the PHI page and the cache is easy to disable.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from fhirground.backends.base import Backend, RawResponse
from fhirground.backends.prompts import RenderedPrompt, render, render_repair
from fhirground.exceptions import SchemaValidationError
from fhirground.preprocess.chunking import Chunk
from fhirground.schemas.mention import MentionList

#: Markdown fences and leading prose a model may wrap JSON in.
_FENCE = re.compile(r"```(?:json)?\s*(?P<body>.*?)\s*```", re.DOTALL)


@dataclass
class ChunkExtraction:
    """What one chunk produced.

    Attributes:
        chunk_id: Which chunk.
        mentions: The parsed mention list, empty when extraction failed.
        samples: Per-sample mention lists, when ``n > 1``. Used for the
            self-consistency confidence component.
        repairs: Repair attempts consumed. Feeds the ``v`` component.
        responses: The raw responses, for the debug trace.
        error: The final error when every attempt failed.
        cached: Whether the first response came from cache.
    """

    chunk_id: str
    mentions: MentionList = field(default_factory=MentionList)
    samples: list[MentionList] = field(default_factory=list)
    repairs: int = 0
    responses: list[RawResponse] = field(default_factory=list)
    error: str | None = None
    cached: bool = False

    @property
    def ok(self) -> bool:
        """Whether a mention list was recovered."""
        return self.error is None

    @property
    def tokens_in(self) -> int:
        """Prompt tokens across every attempt."""
        return sum(r.tokens_in or 0 for r in self.responses)

    @property
    def tokens_out(self) -> int:
        """Completion tokens across every attempt."""
        return sum(r.tokens_out or 0 for r in self.responses)

    @property
    def latency_s(self) -> float:
        """Summed wall-clock across every attempt."""
        return sum(r.latency_s or 0.0 for r in self.responses)


def strip_to_json(text: str) -> str:
    """Recover the JSON value from a response that may be wrapped.

    Handles a markdown fence and leading or trailing prose. Objects **and**
    arrays are both recognized: looking only for ``{...}`` would reduce a bare
    ``[{...}, {...}]`` -- a common near-miss where the model omits the wrapper
    -- to just its first element, silently dropping every other mention.

    Does **not** attempt to fix malformed JSON. Repairing syntax by guessing
    would change what the model said; the repair loop exists to ask it instead.
    """
    if not text:
        return ""
    fenced = _FENCE.search(text)
    if fenced:
        return fenced.group("body").strip()

    stripped = text.strip()
    candidates: list[tuple[int, int]] = []
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = stripped.find(opener), stripped.rfind(closer)
        if start >= 0 and end > start:
            candidates.append((start, end))
    if not candidates:
        return stripped
    # Whichever bracket opens first is the outer value.
    start, end = min(candidates, key=lambda pair: pair[0])
    return stripped[start : end + 1]


def parse_mentions(text: str) -> MentionList:
    """Parse a raw response into a validated :class:`MentionList`.

    Raises:
        SchemaValidationError: With a message written to be read *by a model*
            -- it names the offending field and what was expected, because
            that message is fed straight back in the repair prompt.
    """
    body = strip_to_json(text)
    if not body:
        raise SchemaValidationError(
            "The response was empty. Return a JSON object with a 'mentions' array.",
            raw_excerpt="",
        )

    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise SchemaValidationError(
            f"The response is not valid JSON: {exc.msg} at line {exc.lineno} "
            f"column {exc.colno}. Return a single JSON object and nothing else.",
            raw_excerpt=body[:200],
        ) from exc

    if isinstance(data, list):
        # A bare array is a common near-miss; accept it rather than spending a
        # repair attempt on a wrapper the model clearly intended.
        data = {"mentions": data}
    if not isinstance(data, dict):
        raise SchemaValidationError(
            f"The top level must be a JSON object with a 'mentions' array, but it was "
            f"a {type(data).__name__}.",
            raw_excerpt=body[:200],
        )

    try:
        return MentionList.model_validate(data)
    except ValidationError as exc:
        raise SchemaValidationError(_format_for_model(exc), raw_excerpt=body[:200]) from exc


def _format_for_model(exc: ValidationError) -> str:
    """Render a pydantic error as instructions a model can act on."""
    lines: list[str] = []
    for error in exc.errors()[:6]:
        location = ".".join(str(p) for p in error["loc"]) or "<root>"
        message = error["msg"]
        if error["type"] == "extra_forbidden":
            message = (
                "this field is not in the schema. Remove it. Remember that there are "
                "no code fields."
            )
        lines.append(f"- {location}: {message}")
    return "The JSON did not match the schema:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# the on-disk call cache
# ---------------------------------------------------------------------------


class CallCache:
    """SQLite cache of backend calls.

    Keyed by a hash of backend description, prompt, temperature, and sample
    count, so changing any of them is a cache miss rather than a stale hit.

    The cached value includes the prompt, which includes **note text**. The PHI
    documentation says so, and ``PipelineConfig.cache_path=None`` disables it.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS llm_cache ("
                " key TEXT PRIMARY KEY, payload TEXT NOT NULL,"
                " created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=15.0)

    @staticmethod
    def make_key(
        backend_description: dict[str, Any],
        prompt: str,
        *,
        temperature: float,
        n: int,
        schema_digest: str,
    ) -> str:
        """Hash the inputs that determine a response."""
        blob = json.dumps(
            {
                "backend": backend_description,
                "prompt": prompt,
                "temperature": temperature,
                "n": n,
                "schema": schema_digest,
            },
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, key: str) -> list[RawResponse] | None:
        """Look up cached responses."""
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT payload FROM llm_cache WHERE key = ?", (key,)).fetchone()
        if not row:
            self.misses += 1
            return None
        self.hits += 1
        return [RawResponse(**{**item, "cached": True}) for item in json.loads(row[0])]

    def put(self, key: str, responses: list[RawResponse]) -> None:
        """Store responses."""
        payload = json.dumps(
            [
                {
                    "text": r.text,
                    "model_id": r.model_id,
                    "tokens_in": r.tokens_in,
                    "tokens_out": r.tokens_out,
                    "latency_s": r.latency_s,
                    "logprob": r.logprob,
                    "finish_reason": r.finish_reason,
                }
                for r in responses
            ]
        )
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO llm_cache (key, payload) VALUES (?, ?)",
                (key, payload),
            )

    def stats(self) -> dict[str, int]:
        """Hit and miss counts for the run summary."""
        return {"hits": self.hits, "misses": self.misses}


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def extract_chunk(
    chunk: Chunk,
    backend: Backend,
    schema: dict[str, Any],
    *,
    note_datetime: str | None = None,
    temperature: float = 0.0,
    samples: int = 1,
    max_repairs: int = 2,
    cache: CallCache | None = None,
    schema_digest: str = "",
) -> ChunkExtraction:
    """Extract mentions from one chunk, repairing on schema failure.

    Args:
        chunk: The chunk to extract from.
        backend: The backend to call.
        schema: The mention JSON Schema.
        note_datetime: Passed to the prompt as context.
        temperature: Sampling temperature.
        samples: Number of samples. Above 1 with a non-zero temperature, the
            per-sample lists feed the self-consistency component.
        max_repairs: Attempts after the first failure.
        cache: Optional on-disk cache.
        schema_digest: Short schema hash, part of the cache key.

    Returns:
        A :class:`ChunkExtraction`. A chunk that never parsed comes back with
        ``error`` set and no mentions, rather than raising -- one bad chunk
        should not abandon the note.
    """
    prompt = render(chunk.text, section=chunk.section, note_datetime=note_datetime)
    result = ChunkExtraction(chunk_id=chunk.chunk_id)
    current = prompt
    last_error: str | None = None

    for attempt in range(max_repairs + 1):
        responses = _call(
            backend,
            current,
            schema,
            temperature=temperature,
            n=samples,
            cache=cache,
            schema_digest=schema_digest,
        )
        result.responses.extend(responses)
        if attempt == 0 and responses and responses[0].cached:
            result.cached = True

        parsed: list[MentionList] = []
        error: str | None = None
        for response in responses:
            try:
                parsed.append(parse_mentions(response.text))
            except SchemaValidationError as exc:
                error = str(exc)
                if response.truncated:
                    error = (
                        f"{error}\n\nThe response was cut off at the token limit, so the "
                        "JSON was incomplete. Report fewer findings or shorter evidence "
                        "quotes."
                    )

        if parsed:
            result.mentions = parsed[0]
            result.samples = parsed
            result.repairs = attempt
            result.error = None
            return result

        last_error = error or "the response could not be parsed"
        result.repairs = attempt
        if attempt < max_repairs:
            current = render_repair(prompt, last_error)

    result.error = last_error
    return result


def _call(
    backend: Backend,
    prompt: RenderedPrompt,
    schema: dict[str, Any],
    *,
    temperature: float,
    n: int,
    cache: CallCache | None,
    schema_digest: str,
) -> list[RawResponse]:
    """Call the backend, consulting the cache first."""
    description = backend.describe()
    key = None
    if cache is not None:
        key = CallCache.make_key(
            description,
            prompt.combined,
            temperature=temperature,
            n=n,
            schema_digest=schema_digest,
        )
        hit = cache.get(key)
        if hit is not None:
            return hit

    # Backends that take a system message get one; the protocol's minimum
    # signature does not include it, so this is a capability check rather than
    # an assumption.
    try:
        responses = backend.extract(  # type: ignore[call-arg]
            prompt.user, schema, temperature=temperature, n=n, system=prompt.system
        )
    except TypeError:
        responses = backend.extract(prompt.combined, schema, temperature=temperature, n=n)

    if cache is not None and key is not None and responses:
        cache.put(key, responses)
    return responses
