"""Resolving dates and relative phrases against the note's timestamp.

Explicit dates (ISO, ``MM/DD/YYYY``, ``Month D, YYYY``) and a small set of
relative phrases (``today``, ``yesterday``, ``N days/weeks/months ago``).
Everything else resolves to **None**, and None is honest: a wrong
``effectiveDateTime`` silently misdates a clinical finding, and "last winter"
has no defensible answer.

``MM/DD/YYYY`` is read US-style, which is the convention in US clinical notes
and in the Synthea fixtures. The ambiguity with ``DD/MM`` is real, so
:func:`resolve_date` reports which interpretation it used and refuses the
genuinely ambiguous slash dates when ``day_first`` is left unset and both
readings are valid.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta

_MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

_ISO_RE = re.compile(r"\b(?P<y>\d{4})-(?P<m>\d{1,2})-(?P<d>\d{1,2})\b")
_SLASH_RE = re.compile(r"\b(?P<a>\d{1,2})/(?P<b>\d{1,2})/(?P<y>\d{2,4})\b")
_MONTH_NAME_RE = re.compile(
    r"\b(?P<mon>[A-Za-z]{3,9})\.?\s+(?P<d>\d{1,2})(?:st|nd|rd|th)?,?\s+(?P<y>\d{4})\b"
)
_DAY_MONTH_RE = re.compile(
    r"\b(?P<d>\d{1,2})(?:st|nd|rd|th)?\s+(?P<mon>[A-Za-z]{3,9})\.?,?\s+(?P<y>\d{4})\b"
)
_MONTH_YEAR_RE = re.compile(r"\b(?P<mon>[A-Za-z]{3,9})\.?\s+(?P<y>\d{4})\b")
_YEAR_ONLY_RE = re.compile(r"\b(?:in|since|from)\s+(?P<y>(?:19|20)\d{2})\b", re.IGNORECASE)

_RELATIVE_RE = re.compile(
    r"\b(?P<n>\d{1,4}|a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s*"
    r"(?P<unit>day|days|week|weeks|month|months|year|years|yr|yrs)\s+ago\b",
    re.IGNORECASE,
)
_WORD_NUMBERS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
_TODAY_RE = re.compile(r"\b(today|this\s+morning|this\s+afternoon|this\s+evening|tonight)\b", re.I)
_YESTERDAY_RE = re.compile(r"\byesterday\b", re.IGNORECASE)


class DatePrecision:
    """How much of the date was actually stated."""

    DAY = "day"
    MONTH = "month"
    YEAR = "year"


@dataclass(frozen=True, slots=True)
class ResolvedDate:
    """A date recovered from text.

    Attributes:
        value: The resolved date.
        precision: ``day``, ``month``, or ``year`` -- what the text supported.
        strategy: How it was resolved, for the debug trace.
        text: The original phrase.
        inferred: True when the date came from the note timestamp rather than
            from the text. Drives the ``effective-inferred`` extension.
    """

    value: date
    precision: str
    strategy: str
    text: str
    inferred: bool = False

    def to_fhir(self) -> str:
        """FHIR ``date``/``dateTime`` string at the stated precision.

        A month-precision onset becomes ``2019-03``, not ``2019-03-01``.
        Padding to a day fabricates precision the note never had.
        """
        if self.precision == DatePrecision.YEAR:
            return f"{self.value.year:04d}"
        if self.precision == DatePrecision.MONTH:
            return f"{self.value.year:04d}-{self.value.month:02d}"
        return self.value.isoformat()


def _mk(value: date, precision: str, strategy: str, text: str) -> ResolvedDate:
    return ResolvedDate(value=value, precision=precision, strategy=strategy, text=text)


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def resolve_date(
    text: str | None,
    note_datetime: datetime,
    *,
    day_first: bool | None = None,
) -> ResolvedDate | None:
    """Resolve a date phrase against the note's timestamp.

    Args:
        text: The phrase as written, e.g. ``"3 days ago"`` or ``"03/14/2026"``.
        note_datetime: Anchor for relative phrases.
        day_first: Interpretation for ``A/B/YYYY``. ``False`` forces US
            month-first, ``True`` forces day-first, and ``None`` (the default)
            uses month-first but **refuses** dates where both readings are
            valid and different -- better no date than a date six months off.

    Returns:
        A :class:`ResolvedDate`, or None when nothing could be resolved.
    """
    if not text:
        return None
    phrase = text.strip()
    if not phrase:
        return None

    anchor = note_datetime.date()

    iso = _ISO_RE.search(phrase)
    if iso:
        value = _safe_date(int(iso["y"]), int(iso["m"]), int(iso["d"]))
        if value:
            return _mk(value, DatePrecision.DAY, "iso", phrase)

    slash = _SLASH_RE.search(phrase)
    if slash:
        a, b = int(slash["a"]), int(slash["b"])
        year = int(slash["y"])
        if year < 100:
            year += 2000 if year <= (anchor.year % 100) else 1900
        month_first = _safe_date(year, a, b)
        day_first_value = _safe_date(year, b, a)
        if day_first is True and day_first_value:
            return _mk(day_first_value, DatePrecision.DAY, "slash_day_first", phrase)
        if day_first is False and month_first:
            return _mk(month_first, DatePrecision.DAY, "slash_month_first", phrase)
        if day_first is None:
            if month_first and day_first_value and month_first != day_first_value:
                # Genuinely ambiguous: 03/04/2026 could be either.
                return None
            if month_first:
                return _mk(month_first, DatePrecision.DAY, "slash_month_first", phrase)
            if day_first_value:
                return _mk(day_first_value, DatePrecision.DAY, "slash_day_first", phrase)

    named = _MONTH_NAME_RE.search(phrase)
    if named:
        month = _MONTHS.get(named["mon"].lower())
        if month:
            value = _safe_date(int(named["y"]), month, int(named["d"]))
            if value:
                return _mk(value, DatePrecision.DAY, "month_name", phrase)

    day_month = _DAY_MONTH_RE.search(phrase)
    if day_month:
        month = _MONTHS.get(day_month["mon"].lower())
        if month:
            value = _safe_date(int(day_month["y"]), month, int(day_month["d"]))
            if value:
                return _mk(value, DatePrecision.DAY, "day_month_name", phrase)

    relative = _RELATIVE_RE.search(phrase)
    if relative:
        token = relative["n"].lower()
        count = _WORD_NUMBERS.get(token) if not token.isdigit() else int(token)
        if count is not None:
            unit = relative["unit"].lower().rstrip("s")
            if unit == "day":
                return _mk(anchor - timedelta(days=count), DatePrecision.DAY, "relative", phrase)
            if unit == "week":
                return _mk(anchor - timedelta(weeks=count), DatePrecision.DAY, "relative", phrase)
            if unit == "month":
                return _mk(_shift_months(anchor, -count), DatePrecision.MONTH, "relative", phrase)
            if unit in ("year", "yr"):
                return _mk(
                    _shift_months(anchor, -12 * count), DatePrecision.YEAR, "relative", phrase
                )

    if _TODAY_RE.search(phrase):
        return _mk(anchor, DatePrecision.DAY, "today", phrase)
    if _YESTERDAY_RE.search(phrase):
        return _mk(anchor - timedelta(days=1), DatePrecision.DAY, "yesterday", phrase)

    month_year = _MONTH_YEAR_RE.search(phrase)
    if month_year:
        month = _MONTHS.get(month_year["mon"].lower())
        if month:
            value = _safe_date(int(month_year["y"]), month, 1)
            if value:
                return _mk(value, DatePrecision.MONTH, "month_year", phrase)

    year_only = _YEAR_ONLY_RE.search(phrase)
    if year_only:
        value = _safe_date(int(year_only["y"]), 1, 1)
        if value:
            return _mk(value, DatePrecision.YEAR, "year_only", phrase)

    return None


def _shift_months(anchor: date, months: int) -> date:
    """Shift a date by whole calendar months, clamping the day."""
    total = (anchor.year * 12 + anchor.month - 1) + months
    year, month = divmod(total, 12)
    month += 1
    day = min(anchor.day, _days_in_month(year, month))
    return date(year, month, day)


def _days_in_month(year: int, month: int) -> int:
    import calendar

    return calendar.monthrange(year, month)[1]


def inferred_from_note(note_datetime: datetime) -> ResolvedDate:
    """The note timestamp, marked as inferred.

    Used for an Observation with no resolvable date, so the resource carries a
    usable ``effectiveDateTime`` while the ``effective-inferred`` extension
    records that fhirground supplied it rather than the note.
    """
    return ResolvedDate(
        value=note_datetime.date(),
        precision=DatePrecision.DAY,
        strategy="note_datetime",
        text="",
        inferred=True,
    )
