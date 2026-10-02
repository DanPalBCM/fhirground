"""Parsing numeric values, comparators, and simple ranges.

The cases that actually appear in notes: ``9.2``, ``<0.01``, ``>= 140``,
``120/80``, ``5-10``. Anything more exotic is left as text rather than guessed
at, because a wrong number in a FHIR Observation is worse than a string.

Ranges are a deliberate compromise, specified in SPEC §3.8: the low bound
becomes the value and a note records the original range. A FHIR
``valueQuantity`` holds one number, and silently discarding the upper bound
without saying so would be the kind of quiet lossiness this project is meant to
avoid.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from fhirground.schemas.mention import ValueComparator

#: `<`, `<=`, `>`, `>=`, and the unicode forms clinicians paste in.
_COMPARATORS: dict[str, ValueComparator] = {
    "<": ValueComparator.LT,
    "<=": ValueComparator.LE,
    "≤": ValueComparator.LE,
    ">": ValueComparator.GT,
    ">=": ValueComparator.GE,
    "≥": ValueComparator.GE,
}

_NUMBER = r"[-+]?\d{1,12}(?:[.,]\d{1,6})?"

_COMPARATOR_RE = re.compile(rf"^\s*(?P<cmp><=|>=|≤|≥|<|>)\s*(?P<num>{_NUMBER})\s*$")
_PLAIN_RE = re.compile(rf"^\s*(?P<num>{_NUMBER})\s*$")
_RANGE_RE = re.compile(
    rf"^\s*(?P<low>{_NUMBER})\s*(?:-|\u2013|\u2014|to)\s*(?P<high>{_NUMBER})\s*$",
    re.IGNORECASE,
)
#: Blood pressure, which is written as a pair but is two observations. v1 keeps
#: it as text rather than inventing systolic/diastolic components.
_BP_RE = re.compile(rf"^\s*(?P<sys>{_NUMBER})\s*/\s*(?P<dia>{_NUMBER})\s*$")


@dataclass(frozen=True, slots=True)
class ParsedValue:
    """A parsed observation or dose value.

    Attributes:
        number: The numeric value, or None when the text did not parse.
        comparator: A FHIR ``Quantity.comparator``, when one was written.
        text: The original text, always preserved.
        is_range: True when the input was a range and ``number`` is its low
            bound.
        range_high: The upper bound of a range, kept for the note.
        note: Human-readable caveat, carried into ``Observation.note`` and the
            debug trace. Non-empty exactly when information was dropped.
        kind: ``number``, ``comparator``, ``range``, ``pair``, or ``text``.
    """

    number: float | None = None
    comparator: ValueComparator | None = None
    text: str = ""
    is_range: bool = False
    range_high: float | None = None
    note: str = ""
    kind: str = "text"

    @property
    def is_numeric(self) -> bool:
        """Whether a number was recovered."""
        return self.number is not None


def _to_float(raw: str) -> float | None:
    """Parse a number, tolerating a comma decimal separator."""
    try:
        return float(raw.replace(",", "."))
    except (TypeError, ValueError):
        return None


def parse_value(raw: str | float | int | None) -> ParsedValue | None:
    """Parse a value expression from a mention.

    Args:
        raw: The value as written, or an already-numeric value.

    Returns:
        A :class:`ParsedValue`, or None for an absent or blank input.
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return ParsedValue(number=float(raw), text=str(raw), kind="number")

    text = str(raw).strip()
    if not text:
        return None

    plain = _PLAIN_RE.match(text)
    if plain:
        value = _to_float(plain.group("num"))
        if value is not None:
            return ParsedValue(number=value, text=text, kind="number")

    comparator = _COMPARATOR_RE.match(text)
    if comparator:
        value = _to_float(comparator.group("num"))
        if value is not None:
            return ParsedValue(
                number=value,
                comparator=_COMPARATORS[comparator.group("cmp")],
                text=text,
                kind="comparator",
            )

    ranged = _RANGE_RE.match(text)
    if ranged:
        low, high = _to_float(ranged.group("low")), _to_float(ranged.group("high"))
        if low is not None and high is not None and low <= high:
            return ParsedValue(
                number=low,
                text=text,
                is_range=True,
                range_high=high,
                note=(
                    f"Reported as a range {text!r}; the low bound is recorded as the "
                    "value and the upper bound is not represented in valueQuantity."
                ),
                kind="range",
            )

    pair = _BP_RE.match(text)
    if pair:
        return ParsedValue(
            text=text,
            note=(
                f"Reported as a pair {text!r}, which is two measurements. v1 keeps it "
                "as valueString rather than splitting it into components."
            ),
            kind="pair",
        )

    return ParsedValue(text=text, kind="text")


def parse_dose(dose_value: float | None, dose_unit: str | None) -> tuple[float, str] | None:
    """Validate a dose pair for ``doseAndRate[0].doseQuantity``.

    Returns:
        ``(value, unit_text)`` only when both are present and the value is a
        finite, non-negative number. A dose without a unit is meaningless and a
        unit without a value is not a dose, so either alone yields None and the
        information survives in ``dosageInstruction.text``.
    """
    if dose_value is None or dose_unit is None:
        return None
    unit = dose_unit.strip()
    if not unit:
        return None
    try:
        value = float(dose_value)
    except (TypeError, ValueError):
        return None
    if value < 0 or value != value or value in (float("inf"), float("-inf")):
        return None
    return value, unit
