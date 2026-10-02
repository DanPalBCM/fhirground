"""Units, values, and dates."""

from __future__ import annotations

from datetime import datetime

import pytest

from fhirground.normalize import (
    UCUM_SYSTEM,
    is_known_ucum_code,
    known_unit_count,
    parse_dose,
    parse_value,
    resolve_date,
    resolve_unit,
)
from fhirground.normalize.dates import DatePrecision, inferred_from_note

ANCHOR = datetime(2026, 3, 14, 10, 30)


class TestUnits:
    def test_table_size_is_about_sixty(self):
        assert 50 <= known_unit_count() <= 80

    @pytest.mark.parametrize(
        ("raw", "code"),
        [
            ("mg/dL", "mg/dL"),
            ("mg/dl", "mg/dL"),
            ("milligrams per deciliter", "mg/dL"),
            ("%", "%"),
            ("percent", "%"),
            ("mmHg", "mm[Hg]"),
            ("bpm", "/min"),
            ("beats/min", "/min"),
            ("kg/m2", "kg/m2"),
            ("kg/m^2", "kg/m2"),
            ("mcg", "ug"),
            ("micrograms", "ug"),
            ("mmol/L", "mmol/L"),
            ("U/L", "U/L"),
            ("lbs", "[lb_av]"),
            ("degC", "Cel"),
        ],
    )
    def test_known_units_resolve(self, raw, code):
        r = resolve_unit(raw)
        assert r.code == code
        assert r.system == UCUM_SYSTEM
        assert r.is_ucum

    def test_unknown_unit_is_text_only(self):
        """The rule that stops a guessed code wearing a standards URL."""
        r = resolve_unit("furlongs per fortnight")
        assert r.text == "furlongs per fortnight"
        assert r.code is None
        assert r.system is None
        assert not r.is_ucum

    def test_blank_and_none(self):
        assert resolve_unit(None) is None
        assert resolve_unit("   ") is None

    def test_original_text_is_always_preserved(self):
        assert resolve_unit("mg/dl").text == "mg/dl"

    def test_known_code_check(self):
        assert is_known_ucum_code("mg/dL")
        assert not is_known_ucum_code("made-up")
        assert not is_known_ucum_code(None)


class TestValues:
    def test_plain_number(self):
        p = parse_value("9.2")
        assert p.number == pytest.approx(9.2)
        assert p.kind == "number"
        assert not p.note

    def test_numeric_input(self):
        assert parse_value(88).number == 88.0
        assert parse_value(9.2).number == pytest.approx(9.2)

    @pytest.mark.parametrize(
        ("raw", "cmp", "num"),
        [("<0.01", "<", 0.01), ("<= 5", "<=", 5.0), (">140", ">", 140.0), (">= 7.5", ">=", 7.5)],
    )
    def test_comparators(self, raw, cmp, num):
        p = parse_value(raw)
        assert p.comparator.value == cmp
        assert p.number == pytest.approx(num)
        assert p.kind == "comparator"

    def test_unicode_comparators(self):
        assert parse_value("≤5").comparator.value == "<="
        assert parse_value("≥5").comparator.value == ">="

    def test_range_takes_the_low_bound_and_says_so(self):
        p = parse_value("5-10")
        assert p.number == 5.0
        assert p.range_high == 10.0
        assert p.is_range
        assert "upper bound is not represented" in p.note

    def test_range_with_words(self):
        assert parse_value("5 to 10").number == 5.0

    def test_pair_stays_text_with_a_note(self):
        """120/80 is two measurements; splitting it would invent components."""
        p = parse_value("120/80")
        assert p.number is None
        assert p.kind == "pair"
        assert "two measurements" in p.note

    def test_comma_decimal(self):
        assert parse_value("9,2").number == pytest.approx(9.2)

    def test_unparseable_stays_text(self):
        p = parse_value("trace")
        assert p.number is None
        assert p.kind == "text"
        assert p.text == "trace"

    def test_blank_and_none(self):
        assert parse_value(None) is None
        assert parse_value("  ") is None


class TestDose:
    def test_valid_pair(self):
        assert parse_dose(500, "mg") == (500.0, "mg")

    @pytest.mark.parametrize(
        ("value", "unit"), [(None, "mg"), (500, None), (500, "  "), (-1, "mg")]
    )
    def test_incomplete_or_invalid_is_none(self, value, unit):
        assert parse_dose(value, unit) is None

    def test_zero_dose_is_allowed(self):
        assert parse_dose(0, "mg") == (0.0, "mg")


class TestDates:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("2019-03-07", "2019-03-07"),
            ("March 7, 2019", "2019-03-07"),
            ("7 March 2019", "2019-03-07"),
            ("Mar 7, 2019", "2019-03-07"),
        ],
    )
    def test_explicit_dates(self, text, expected):
        assert resolve_date(text, ANCHOR).to_fhir() == expected

    def test_us_slash_order(self):
        assert resolve_date("03/14/2026", ANCHOR).to_fhir() == "2026-03-14"

    def test_ambiguous_slash_date_is_refused(self):
        """Better no date than one six months off."""
        assert resolve_date("03/04/2026", ANCHOR) is None

    def test_day_first_can_be_forced(self):
        assert resolve_date("03/04/2026", ANCHOR, day_first=True).to_fhir() == "2026-04-03"
        assert resolve_date("03/04/2026", ANCHOR, day_first=False).to_fhir() == "2026-03-04"

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("today", "2026-03-14"),
            ("this morning", "2026-03-14"),
            ("yesterday", "2026-03-13"),
            ("3 days ago", "2026-03-11"),
            ("two weeks ago", "2026-02-28"),
            ("1 week ago", "2026-03-07"),
        ],
    )
    def test_relative_phrases(self, text, expected):
        assert resolve_date(text, ANCHOR).to_fhir() == expected

    def test_months_ago_keeps_month_precision(self):
        r = resolve_date("6 months ago", ANCHOR)
        assert r.precision == DatePrecision.MONTH
        assert r.to_fhir() == "2025-09"

    def test_years_ago_keeps_year_precision(self):
        r = resolve_date("2 years ago", ANCHOR)
        assert r.precision == DatePrecision.YEAR
        assert r.to_fhir() == "2024"

    def test_month_year_precision(self):
        r = resolve_date("March 2019", ANCHOR)
        assert r.precision == DatePrecision.MONTH
        assert r.to_fhir() == "2019-03"

    def test_year_only_precision(self):
        r = resolve_date("since 2019", ANCHOR)
        assert r.precision == DatePrecision.YEAR
        assert r.to_fhir() == "2019"

    @pytest.mark.parametrize(
        "text", ["last winter", "a while back", "recently", "", "   ", "sometime"]
    )
    def test_unresolvable_is_none(self, text):
        assert resolve_date(text, ANCHOR) is None

    def test_none_input(self):
        assert resolve_date(None, ANCHOR) is None

    def test_invalid_calendar_date_is_rejected(self):
        assert resolve_date("2019-02-30", ANCHOR) is None

    def test_inferred_from_note_is_flagged(self):
        r = inferred_from_note(ANCHOR)
        assert r.inferred is True
        assert r.to_fhir() == "2026-03-14"
        assert r.strategy == "note_datetime"

    def test_explicit_dates_are_not_flagged_inferred(self):
        assert resolve_date("today", ANCHOR).inferred is False
