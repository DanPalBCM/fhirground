"""Local value-set resolvers: user-supplied CSV or JSON.

This is the resolver most deployments will use, because the real terminologies
cannot be bundled. SNOMED CT, LOINC, and RxNorm each carry licensing
obligations, so fhirground ships only tiny synthetic value sets for its tests
and documentation, and the docs say plainly that a real deployment supplies its
own.

Matching is two-stage: normalized exact match on the display or any synonym,
then rapidfuzz ``token_set_ratio`` above a threshold. Token-set is the right
choice for clinical text because it ignores word order and duplication --
"diabetes mellitus type 2" and "type 2 diabetes mellitus" are the same concept
and a plain ratio would score them poorly.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz, process

from fhirground.exceptions import TerminologyError
from fhirground.schemas.mention import MentionType
from fhirground.terminology.base import Coding, Resolution

#: Default minimum token_set_ratio for a fuzzy value-set hit.
DEFAULT_FUZZY_THRESHOLD = 88.0

#: Columns a value-set CSV must provide.
REQUIRED_COLUMNS = ("system", "code", "display")


@dataclass(frozen=True, slots=True)
class ValueSetEntry:
    """One concept in a value set."""

    system: str
    code: str
    display: str
    synonyms: tuple[str, ...] = field(default_factory=tuple)

    @property
    def surface_forms(self) -> tuple[str, ...]:
        """Display plus synonyms, in preference order."""
        return (self.display, *self.synonyms)

    def to_coding(self) -> Coding:
        """The code this entry represents."""
        return Coding(system=self.system, code=self.code, display=self.display)


def _normalize(text: str) -> str:
    """Lower-case, collapse whitespace, strip trailing punctuation."""
    return " ".join(text.lower().split()).strip(" .,;:")


class LocalValueSet:
    """A resolver backed by an in-memory value set.

    Example::

        conditions = LocalValueSet.from_csv("valuesets/conditions.csv")
        conditions.resolve("type 2 diabetes", MentionType.CONDITION)
    """

    def __init__(
        self,
        entries: list[ValueSetEntry],
        *,
        name: str = "local",
        fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
        fuzzy: bool = True,
    ) -> None:
        if not entries:
            raise TerminologyError(
                f"value set {name!r} is empty; a resolver with no concepts can only "
                "ever return text-only resolutions, so use NullResolver instead"
            )
        self.name = name
        self.entries = entries
        self.fuzzy_threshold = fuzzy_threshold
        self.fuzzy = fuzzy

        # Surface form -> entry. Built once; the first entry wins a collision so
        # value-set order is the tie-break and the result is deterministic.
        self._index: dict[str, ValueSetEntry] = {}
        for entry in entries:
            for form in entry.surface_forms:
                key = _normalize(form)
                if key and key not in self._index:
                    self._index[key] = entry
        self._keys = list(self._index)

    def __len__(self) -> int:
        return len(self.entries)

    # -- construction ---------------------------------------------------

    @classmethod
    def from_csv(cls, path: str | Path, *, name: str | None = None, **kwargs: Any) -> LocalValueSet:
        """Load from a CSV with ``system,code,display,synonyms``.

        ``synonyms`` is pipe-separated, as SPEC §6.3 specifies.
        """
        path = Path(path)
        if not path.exists():
            raise TerminologyError(f"value set file not found: {path}")

        entries: list[ValueSetEntry] = []
        with path.open(encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                raise TerminologyError(f"value set {path} has no header row")
            missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
            if missing:
                raise TerminologyError(
                    f"value set {path} is missing column(s) {missing}; "
                    f"required: {list(REQUIRED_COLUMNS)}, found: {reader.fieldnames}"
                )
            for lineno, row in enumerate(reader, start=2):
                system = (row.get("system") or "").strip()
                code = (row.get("code") or "").strip()
                display = (row.get("display") or "").strip()
                if not (system and code and display):
                    raise TerminologyError(
                        f"{path}:{lineno}: system, code, and display are all required"
                    )
                raw_syn = (row.get("synonyms") or "").strip()
                synonyms = tuple(s.strip() for s in raw_syn.split("|") if s.strip())
                entries.append(
                    ValueSetEntry(system=system, code=code, display=display, synonyms=synonyms)
                )
        return cls(entries, name=name or f"local:{path.stem}", **kwargs)

    @classmethod
    def from_json(
        cls, path: str | Path, *, name: str | None = None, **kwargs: Any
    ) -> LocalValueSet:
        """Load from a JSON list of ``{system, code, display, synonyms}``."""
        path = Path(path)
        if not path.exists():
            raise TerminologyError(f"value set file not found: {path}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise TerminologyError(f"{path}: malformed JSON: {exc.msg}") from exc
        if not isinstance(data, list):
            raise TerminologyError(f"{path}: expected a JSON array of concepts")

        entries = []
        for i, row in enumerate(data):
            if not isinstance(row, dict):
                raise TerminologyError(f"{path}: entry {i} is not an object")
            missing = [c for c in REQUIRED_COLUMNS if not row.get(c)]
            if missing:
                raise TerminologyError(f"{path}: entry {i} is missing {missing}")
            synonyms = row.get("synonyms") or []
            if isinstance(synonyms, str):
                synonyms = [s.strip() for s in synonyms.split("|") if s.strip()]
            entries.append(
                ValueSetEntry(
                    system=str(row["system"]),
                    code=str(row["code"]),
                    display=str(row["display"]),
                    synonyms=tuple(str(s) for s in synonyms),
                )
            )
        return cls(entries, name=name or f"local:{path.stem}", **kwargs)

    @classmethod
    def from_entries(
        cls, rows: list[dict[str, Any]], *, name: str = "local", **kwargs: Any
    ) -> LocalValueSet:
        """Build from plain dicts, for tests and programmatic use."""
        entries = []
        for row in rows:
            synonyms = row.get("synonyms") or ()
            if isinstance(synonyms, str):
                synonyms = tuple(s.strip() for s in synonyms.split("|") if s.strip())
            entries.append(
                ValueSetEntry(
                    system=str(row["system"]),
                    code=str(row["code"]),
                    display=str(row["display"]),
                    synonyms=tuple(synonyms),
                )
            )
        return cls(entries, name=name, **kwargs)

    # -- resolution -----------------------------------------------------

    def resolve(
        self,
        text: str,
        mention_type: MentionType,
        context: dict[str, Any] | None = None,
    ) -> Resolution | None:
        """Resolve text against this value set.

        Returns:
            A coded :class:`Resolution` on a hit, or None so a chain can try
            the next resolver. Scores: 1.0 for an exact display match, 0.95 for
            an exact synonym match, and ``ratio/100`` for a fuzzy hit.
        """
        _ = mention_type, context
        key = _normalize(text or "")
        if not key:
            return None

        entry = self._index.get(key)
        if entry is not None:
            exact_display = _normalize(entry.display) == key
            return Resolution(
                text=text,
                coding=entry.to_coding(),
                score=1.0 if exact_display else 0.95,
                resolver=self.name,
                method="exact" if exact_display else "synonym",
            )

        if not self.fuzzy:
            return None

        matches = process.extract(key, self._keys, scorer=fuzz.token_set_ratio, limit=3)
        if not matches:
            return None
        best_key, best_score, _ = matches[0]
        if best_score < self.fuzzy_threshold:
            return None

        entry = self._index[best_key]
        return Resolution(
            text=text,
            coding=entry.to_coding(),
            score=round(best_score / 100.0, 4),
            resolver=self.name,
            method="fuzzy",
            candidates=tuple((m[0], round(m[1] / 100.0, 4)) for m in matches[1:]),
        )

    def known_codes(self) -> set[tuple[str, str]]:
        """Every ``(system, code)`` in this value set."""
        return {(e.system, e.code) for e in self.entries}


class NullResolver:
    """A resolver that never resolves anything.

    Not a placeholder: it is the correct choice when no licensed terminology is
    available. Every resource becomes a text-only ``CodeableConcept``, the
    terminology confidence component is absent rather than zero, and the output
    is honest about carrying no codes.
    """

    name = "null"

    def resolve(
        self,
        text: str,
        mention_type: MentionType,
        context: dict[str, Any] | None = None,
    ) -> Resolution | None:
        """Always returns None, so the mention stays text-only."""
        _ = text, mention_type, context
        return None

    def known_codes(self) -> set[tuple[str, str]]:
        """Empty: this resolver can emit no codes at all."""
        return set()
