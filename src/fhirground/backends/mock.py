"""The mock backend: deterministic, offline, fixture-driven.

Used by every unit test and by the bundled demo, which is how the whole
pipeline runs in CI with no network and no API key. It is not a toy: the demo's
end-to-end guarantee depends on it behaving exactly like a real backend from
``extract()`` outward.

Two modes:

* **fixtures** -- a mapping from chunk text (or a hash of it) to the mention
  JSON a model returned. Recorded responses, replayed exactly.
* **rules** -- a small deterministic extractor over the note text, so a
  freshly generated synthetic note produces sensible mentions without anyone
  hand-writing a fixture for it.

The rule mode is deliberately simple-minded and makes the same mistakes a weak
model makes, which keeps the drop counters and confidence components exercised
rather than always perfect.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from fhirground.backends.base import RawResponse
from fhirground.exceptions import BackendError

#: Phrases that mark the following finding as negated.
_NEGATION = re.compile(
    r"\b(denies|denied|no\s+(?:evidence\s+of\s+|signs?\s+of\s+|complaints?\s+of\s+)?|"
    r"without|negative\s+for|rules?\s+out|ruled\s+out|free\s+of)\b",
    re.IGNORECASE,
)
#: Phrases that mark a finding as hedged.
_HEDGE = re.compile(r"\b(possible|possibly|probable|suspected|likely|rule\s+out|r/o)\b", re.I)
#: Phrases that mark a finding as somebody else's.
_FAMILY = re.compile(
    r"\b(mother|father|mom|dad|sister|brother|sibling|parent|grandmother|grandfather|"
    r"family\s+history|maternal|paternal)\b",
    re.IGNORECASE,
)
#: Phrases that mark a finding as past.
_HISTORICAL = re.compile(r"\b(history\s+of|h/o|previously|in\s+the\s+past|former)\b", re.I)
#: Phrases that mark a finding as conditional.
_HYPOTHETICAL = re.compile(
    r"\b(if\s+(?:he|she|they|the\s+patient)|should\s+(?:he|she|they)|"
    r"in\s+case\s+of|watch\s+for)\b",
    re.IGNORECASE,
)

_START = re.compile(r"\b(start|begin|initiate|commenc\w+|new\s+prescription)\b", re.I)
_STOP = re.compile(r"\b(stop|discontinu\w+|d/c|held|hold)\b", re.IGNORECASE)
_CONTINUE = re.compile(r"\b(continue|cont\.|maintain|unchanged)\b", re.IGNORECASE)
_CHANGE = re.compile(r"\b(increase|decrease|titrat\w+|adjust|chang\w+)\b", re.IGNORECASE)

#: Condition phrases the rule extractor recognizes, longest first so
#: "type 2 diabetes" wins over "diabetes".
_CONDITIONS: tuple[str, ...] = (
    "type 2 diabetes mellitus",
    "type 1 diabetes mellitus",
    "type 2 diabetes",
    "type 1 diabetes",
    "diabetes mellitus",
    "chronic kidney disease",
    "congestive heart failure",
    "coronary artery disease",
    "atrial fibrillation",
    "hyperlipidemia",
    "dyslipidemia",
    "hypertension",
    "asthma",
    "copd",
    "pneumonia",
    "chest pain",
    "obesity",
    "anemia",
    "depression",
    "diabetes",
    "migraine",
    "osteoarthritis",
    "hypothyroidism",
    "gerd",
)
#: Medication phrases.
_MEDICATIONS: tuple[str, ...] = (
    "metformin hydrochloride",
    "lisinopril",
    "metformin",
    "atorvastatin",
    "amlodipine",
    "metoprolol",
    "levothyroxine",
    "albuterol",
    "omeprazole",
    "simvastatin",
    "losartan",
    "gabapentin",
    "sertraline",
    "insulin glargine",
    "hydrochlorothiazide",
    "aspirin",
    "warfarin",
    "prednisone",
    "furosemide",
)
#: Observation names paired with the category they belong to.
_OBSERVATIONS: tuple[tuple[str, str], ...] = (
    ("hemoglobin a1c", "laboratory"),
    ("hba1c", "laboratory"),
    ("a1c", "laboratory"),
    ("ldl cholesterol", "laboratory"),
    ("ldl", "laboratory"),
    ("hdl cholesterol", "laboratory"),
    ("total cholesterol", "laboratory"),
    ("creatinine", "laboratory"),
    ("egfr", "laboratory"),
    ("potassium", "laboratory"),
    ("sodium", "laboratory"),
    ("hemoglobin", "laboratory"),
    ("tsh", "laboratory"),
    ("blood pressure", "vital-signs"),
    ("heart rate", "vital-signs"),
    ("pulse", "vital-signs"),
    ("temperature", "vital-signs"),
    ("weight", "vital-signs"),
    ("height", "vital-signs"),
    ("bmi", "vital-signs"),
    ("respiratory rate", "vital-signs"),
    ("oxygen saturation", "vital-signs"),
)

_DOSE = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>mcg|mg|g|units?|mL|ml|tablets?|puffs?)\b", re.I
)
_ROUTE = re.compile(r"\b(PO|IV|IM|SubQ|SC|topical|inhaled|by mouth|orally)\b", re.IGNORECASE)
_FREQ = re.compile(
    r"\b(BID|TID|QID|QD|daily|twice daily|three times daily|q\d+h|PRN|at bedtime|qhs)\b", re.I
)
_VALUE = re.compile(
    r"(?P<cmp><=|>=|<|>)?\s*(?P<num>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>%|mg/dL|mg/dl|mmol/L|mmHg|bpm|kg|lbs?|cm|in|mL/min|mEq/L|k/uL|ng/mL|°?[CF])?",
    re.IGNORECASE,
)


class MockBackend:
    """A deterministic backend for tests, demos, and offline runs.

    Args:
        fixtures: Chunk text (or its hash) -> mention JSON. Checked first.
        rules: Fall back to the rule extractor when no fixture matches. With
            both off, an unmatched chunk returns no mentions.
        model_id: Recorded in Provenance.
        latency_s: Simulated latency, reported but not slept.
        fail_first_n: Return malformed JSON for the first N calls, to exercise
            the repair loop deterministically.
        malformed_payload: What to return while failing.
    """

    name = "mock"

    def __init__(
        self,
        fixtures: dict[str, Any] | None = None,
        *,
        rules: bool = True,
        model_id: str = "mock-deterministic-v1",
        latency_s: float = 0.0,
        fail_first_n: int = 0,
        malformed_payload: str = '{"mentions": [',
    ) -> None:
        self.fixtures = dict(fixtures or {})
        self.rules = rules
        self.model_id = model_id
        self.latency_s = latency_s
        self.fail_first_n = fail_first_n
        self.malformed_payload = malformed_payload
        self.calls = 0

    # -- construction ---------------------------------------------------

    @classmethod
    def from_fixture_file(cls, path: str | Path, **kwargs: Any) -> MockBackend:
        """Load fixtures from a JSON object mapping chunk text to mention JSON."""
        source = Path(path)
        if not source.exists():
            raise BackendError(f"mock fixture file not found: {source}")
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise BackendError(f"{source}: malformed JSON: {exc.msg}") from exc
        if not isinstance(data, dict):
            raise BackendError(f"{source}: expected a JSON object of chunk -> mentions")
        return cls(fixtures=data, **kwargs)

    @staticmethod
    def fixture_key(text: str) -> str:
        """Stable key for a chunk, so fixtures need not embed note text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    # -- the Backend protocol -------------------------------------------

    def extract(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        temperature: float = 0.0,
        n: int = 1,
    ) -> list[RawResponse]:
        """Return ``n`` deterministic responses.

        Samples are identical regardless of temperature: a mock that varied
        would make tests flaky, and self-consistency is tested directly in
        :mod:`fhirground.dedup` instead.
        """
        _ = schema, temperature
        self.calls += 1

        if self.calls <= self.fail_first_n:
            return [
                RawResponse(
                    text=self.malformed_payload,
                    model_id=self.model_id,
                    latency_s=self.latency_s,
                    finish_reason="stop",
                )
                for _ in range(max(1, n))
            ]

        chunk = _chunk_from_prompt(prompt)
        payload = self._lookup(chunk)
        text = json.dumps(payload)
        return [
            RawResponse(
                text=text,
                model_id=self.model_id,
                tokens_in=len(prompt) // 4,
                tokens_out=len(text) // 4,
                latency_s=self.latency_s,
                finish_reason="stop",
            )
            for _ in range(max(1, n))
        ]

    def describe(self) -> dict[str, Any]:
        """Configuration summary."""
        return {
            "backend": self.name,
            "model_id": self.model_id,
            "n_fixtures": len(self.fixtures),
            "rules": self.rules,
        }

    def _lookup(self, chunk: str) -> dict[str, Any]:
        """Fixture first, then rules, then nothing."""
        if chunk in self.fixtures:
            return _as_payload(self.fixtures[chunk])
        key = self.fixture_key(chunk)
        if key in self.fixtures:
            return _as_payload(self.fixtures[key])
        normalized = " ".join(chunk.split())
        if normalized in self.fixtures:
            return _as_payload(self.fixtures[normalized])
        if self.rules:
            return {"mentions": extract_by_rules(chunk)}
        return {"mentions": []}


def _as_payload(value: Any) -> dict[str, Any]:
    """Accept either a full payload or a bare mention list."""
    if isinstance(value, dict) and "mentions" in value:
        return value
    if isinstance(value, list):
        return {"mentions": value}
    raise BackendError(
        f"mock fixture must be a mention list or an object with a 'mentions' key, "
        f"got {type(value).__name__}"
    )


_NOTE_BLOCK = re.compile(
    r"--- NOTE TEXT START ---\n(?P<chunk>.*?)\n--- NOTE TEXT END ---", re.DOTALL
)


def _chunk_from_prompt(prompt: str) -> str:
    """Recover the chunk text from a rendered prompt.

    The mock needs the chunk, and the prompt is what a backend receives. The
    delimiters are part of the template, so this is a parse of our own format
    rather than a guess.
    """
    match = _NOTE_BLOCK.search(prompt)
    return match.group("chunk") if match else prompt


# ---------------------------------------------------------------------------
# the rule extractor
# ---------------------------------------------------------------------------


def _sentences(text: str) -> list[tuple[str, int]]:
    """Split into sentences with their offsets inside ``text``."""
    out: list[tuple[str, int]] = []
    start = 0
    for match in re.finditer(r"(?<=[.!?;:])\s+|\n+", text):
        piece = text[start : match.start()]
        if piece.strip():
            out.append((piece, start))
        start = match.end()
    tail = text[start:]
    if tail.strip():
        out.append((tail, start))
    return out


def _assertion_for(sentence: str) -> str:
    """Pick an assertion from the sentence's surface cues, in priority order."""
    if _FAMILY.search(sentence):
        return "family"
    if _HYPOTHETICAL.search(sentence):
        return "hypothetical"
    if _NEGATION.search(sentence):
        return "negated"
    if _HEDGE.search(sentence):
        return "possible"
    if _HISTORICAL.search(sentence):
        return "historical"
    return "present"


def _med_action_for(sentence: str) -> str | None:
    if _STOP.search(sentence):
        return "stop"
    if _START.search(sentence):
        return "start"
    if _CHANGE.search(sentence):
        return "change"
    if _CONTINUE.search(sentence):
        return "continue"
    return None


def extract_by_rules(text: str) -> list[dict[str, Any]]:
    """A deterministic, deliberately naive extractor over note text.

    Produces the mention shapes a weak model produces, including the mistakes:
    it reads cues sentence-wide, so a sentence containing both a negation and a
    second finding mislabels the second one. That is on purpose -- it keeps the
    assertion policy and the drop counters exercised by the demo rather than
    always seeing clean input.
    """
    mentions: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for sentence, _offset in _sentences(text):
        lowered = sentence.lower()
        assertion = _assertion_for(sentence)
        evidence = sentence.strip()
        if not evidence:
            continue

        for phrase in _CONDITIONS:
            if phrase in lowered:
                key = ("condition", phrase, assertion)
                if key in seen:
                    continue
                seen.add(key)
                mention: dict[str, Any] = {
                    "mention_type": "condition",
                    "text": phrase,
                    "evidence": evidence[:300],
                    "assertion": assertion,
                }
                if assertion == "historical":
                    mention["clinical_status_hint"] = "resolved"
                mentions.append(mention)
                break

        for phrase in _MEDICATIONS:
            if phrase in lowered:
                key = ("medication", phrase, assertion)
                if key in seen:
                    continue
                seen.add(key)
                mention = {
                    "mention_type": "medication",
                    "text": phrase,
                    "evidence": evidence[:300],
                    "assertion": assertion,
                    "med_action": _med_action_for(sentence),
                }
                dose = _DOSE.search(sentence)
                if dose:
                    mention["dose_value"] = float(dose.group("value"))
                    mention["dose_unit"] = dose.group("unit")
                route = _ROUTE.search(sentence)
                if route:
                    mention["route"] = route.group(1)
                freq = _FREQ.search(sentence)
                if freq:
                    mention["frequency_text"] = freq.group(1)
                mentions.append(mention)
                break

        for phrase, category in _OBSERVATIONS:
            if phrase not in lowered:
                continue
            key = ("observation", phrase, assertion)
            if key in seen:
                continue
            seen.add(key)
            mention = {
                "mention_type": "observation",
                "text": phrase,
                "evidence": evidence[:300],
                "assertion": assertion,
                "obs_category": category,
            }
            after = lowered.split(phrase, 1)[1] if phrase in lowered else ""
            value = _VALUE.search(after[:40])
            if value:
                mention["value_number"] = float(value.group("num"))
                if value.group("cmp"):
                    mention["value_comparator"] = value.group("cmp")
                if value.group("unit"):
                    mention["value_unit"] = value.group("unit")
            for cue in ("today", "yesterday"):
                if cue in lowered:
                    mention["date_text"] = cue
                    break
            mentions.append(mention)
            break

    return mentions


def record_fixtures(chunks: list[str], extractor: Any = None) -> dict[str, list[dict[str, Any]]]:
    """Build a fixture mapping from chunks, for checking in.

    Args:
        chunks: Chunk texts.
        extractor: Callable returning mentions for a chunk. Defaults to the
            rule extractor.

    Returns:
        Hash-keyed fixtures, so the file does not embed note text verbatim.
    """
    fn = extractor or extract_by_rules
    _ = time  # kept for API symmetry with recorded live fixtures
    return {MockBackend.fixture_key(c): fn(c) for c in chunks}
