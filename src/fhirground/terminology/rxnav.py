"""RxNav resolver for medications: optional, opt-in, cached.

Uses the NLM RxNav ``approximateTerm`` endpoint, which is a public service that
needs no API key. It is **off by default** and must be constructed explicitly,
because:

- it makes network calls, so results depend on a remote service and are not
  reproducible across runs;
- it sends **medication text extracted from a clinical note** to a third party.
  That text is a note fragment. Sites with a data-use agreement that forbids
  sending note content outside their boundary must not enable this.

The second point is why :class:`RxNavResolver` takes no default and the docs
carry a warning. A cache is on by default so the same term is not sent twice.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

import httpx

from fhirground.exceptions import TerminologyError
from fhirground.schemas.mention import MentionType
from fhirground.terminology.base import Coding, Resolution

#: RxNorm's code system URI.
RXNORM_SYSTEM = "http://www.nlm.nih.gov/research/umls/rxnorm"

#: The public RxNav base URL.
DEFAULT_BASE_URL = "https://rxnav.nlm.nih.gov/REST"

#: RxNav returns a 0-100 score; this is the minimum fhirground accepts.
DEFAULT_MIN_SCORE = 50.0


class _Cache:
    """A tiny SQLite cache of term -> response.

    On disk so it survives between runs. The docs warn that it contains
    medication text taken from notes, which makes it note-derived data and
    subject to the same handling rules as the notes themselves.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS rxnav_cache ("
                " term TEXT PRIMARY KEY, response TEXT NOT NULL,"
                " fetched_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10.0)

    def get(self, term: str) -> Any | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT response FROM rxnav_cache WHERE term = ?", (term,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, term: str, response: Any) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO rxnav_cache (term, response) VALUES (?, ?)",
                (term, json.dumps(response)),
            )

    def terms(self) -> list[str]:
        with self._lock, self._connect() as conn:
            return [r[0] for r in conn.execute("SELECT term FROM rxnav_cache")]


class RxNavResolver:
    """Resolve medication text to an RxNorm code through NLM RxNav.

    Args:
        base_url: RxNav REST base.
        cache_path: SQLite cache location. None disables caching, which means
            every lookup is a network call.
        min_score: Minimum RxNav approximate-match score to accept.
        timeout: Per-request timeout in seconds.
        client: Inject an ``httpx.Client`` for testing. Tests use ``respx`` to
            mock transport, so no test here touches the network.
        offline: When true, answer only from cache and never call out. Lets a
            cached corpus be re-run with no network at all.

    Example::

        resolver = RxNavResolver(cache_path=".cache/rxnav.sqlite")
        resolver.resolve("metformin 500 mg", MentionType.MEDICATION)
    """

    name = "rxnav"

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        cache_path: str | Path | None = ".cache/rxnav.sqlite",
        min_score: float = DEFAULT_MIN_SCORE,
        timeout: float = 10.0,
        client: httpx.Client | None = None,
        offline: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.min_score = min_score
        self.timeout = timeout
        self.offline = offline
        self._client = client
        self._owns_client = client is None
        self._cache = _Cache(cache_path) if cache_path else None
        self._resolved: set[tuple[str, str]] = set()

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def close(self) -> None:
        """Close the HTTP client if this resolver created it."""
        if self._client is not None and self._owns_client:
            self._client.close()
            self._client = None

    def __enter__(self) -> RxNavResolver:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _fetch(self, term: str) -> Any | None:
        """Fetch an approximate match, preferring the cache."""
        if self._cache is not None:
            cached = self._cache.get(term)
            if cached is not None:
                return cached
        if self.offline:
            return None

        url = f"{self.base_url}/approximateTerm.json"
        try:
            response = self._get_client().get(url, params={"term": term, "maxEntries": 3})
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError as exc:
            raise TerminologyError(
                f"RxNav request failed for a medication term: {type(exc).__name__}. "
                "Set offline=True to run from cache only."
            ) from exc
        except json.JSONDecodeError as exc:
            raise TerminologyError(f"RxNav returned a non-JSON response: {exc.msg}") from exc

        if self._cache is not None:
            self._cache.put(term, payload)
        return payload

    def resolve(
        self,
        text: str,
        mention_type: MentionType,
        context: dict[str, Any] | None = None,
    ) -> Resolution | None:
        """Resolve medication text to an RxNorm code.

        Returns None for non-medication mentions: RxNorm has nothing to say
        about a condition, and querying it anyway would waste a call and risk a
        nonsense match.
        """
        _ = context
        if mention_type is not MentionType.MEDICATION:
            return None
        term = (text or "").strip()
        if not term:
            return None

        payload = self._fetch(term)
        if not payload:
            return None

        candidates = (payload.get("approximateGroup") or {}).get("candidate") or []
        if not candidates:
            return None

        scored: list[tuple[str, str, float]] = []
        for candidate in candidates:
            rxcui = str(candidate.get("rxcui") or "").strip()
            if not rxcui:
                continue
            try:
                score = float(candidate.get("score") or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            scored.append((rxcui, str(candidate.get("name") or "").strip(), score))

        if not scored:
            return None
        scored.sort(key=lambda row: -row[2])
        rxcui, display, score = scored[0]
        if score < self.min_score:
            return None

        self._resolved.add((RXNORM_SYSTEM, rxcui))
        return Resolution(
            text=text,
            coding=Coding(system=RXNORM_SYSTEM, code=rxcui, display=display or None),
            score=round(min(score, 100.0) / 100.0, 4),
            resolver=self.name,
            method="remote",
            candidates=tuple((name or code, round(s / 100.0, 4)) for code, name, s in scored[1:]),
        )

    def known_codes(self) -> set[tuple[str, str]]:
        """Codes this resolver has actually returned, plus anything cached.

        A remote resolver cannot enumerate its terminology in advance, so the
        no-invented-codes test treats remote-resolved codes as a separate,
        explicitly-allowed category rather than pretending this set is complete.
        """
        out = set(self._resolved)
        if self._cache is not None:
            for term in self._cache.terms():
                payload = self._cache.get(term) or {}
                for candidate in (payload.get("approximateGroup") or {}).get("candidate") or []:
                    rxcui = str(candidate.get("rxcui") or "").strip()
                    if rxcui:
                        out.add((RXNORM_SYSTEM, rxcui))
        return out
