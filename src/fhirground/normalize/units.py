"""Common clinical units mapped to UCUM.

A bundled table of roughly sixty units, which covers what appears in vitals and
routine labs. It is **not** a UCUM implementation: no parsing of arbitrary
expressions, no arithmetic on dimensions.

The rule that matters: a unit fhirground does not recognize is kept as
``valueQuantity.unit`` **text only**, with no ``system`` and no ``code``.
Emitting ``http://unitsofmeasure.org`` beside a guessed code would be a
hallucinated code wearing a standards URL, which is precisely what this project
exists to avoid. A project validation rule enforces it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: The UCUM code system URI, used only with codes from :data:`UCUM_TABLE`.
UCUM_SYSTEM = "http://unitsofmeasure.org"

#: Surface form (normalized) -> UCUM code. Keys are lower-cased with
#: whitespace removed; lookup normalizes the input the same way.
#:
#: Sources: the UCUM "common units" list and the units that appear in Synthea's
#: Observation output. Where a unit has several conventional spellings, each
#: spelling is a key -- "mg/dL", "mg/dl", and "milligrams per deciliter" all
#: resolve to ``mg/dL``.
UCUM_TABLE: dict[str, str] = {
    # dimensionless and proportions
    "%": "%",
    "percent": "%",
    "ratio": "1",
    "": "1",
    # mass
    "g": "g",
    "gram": "g",
    "grams": "g",
    "kg": "kg",
    "kilogram": "kg",
    "kilograms": "kg",
    "mg": "mg",
    "milligram": "mg",
    "milligrams": "mg",
    "mcg": "ug",
    "ug": "ug",
    "microgram": "ug",
    "micrograms": "ug",
    "ng": "ng",
    "nanogram": "ng",
    "lb": "[lb_av]",
    "lbs": "[lb_av]",
    "pound": "[lb_av]",
    "pounds": "[lb_av]",
    "oz": "[oz_av]",
    # length
    "m": "m",
    "meter": "m",
    "meters": "m",
    "cm": "cm",
    "centimeter": "cm",
    "centimeters": "cm",
    "mm": "mm",
    "millimeter": "mm",
    "in": "[in_i]",
    "inch": "[in_i]",
    "inches": "[in_i]",
    "ft": "[ft_i]",
    # volume
    "l": "L",
    "liter": "L",
    "liters": "L",
    "dl": "dL",
    "deciliter": "dL",
    "ml": "mL",
    "milliliter": "mL",
    "milliliters": "mL",
    "ul": "uL",
    "microliter": "uL",
    # concentration
    "mg/dl": "mg/dL",
    "milligramsperdeciliter": "mg/dL",
    "g/dl": "g/dL",
    "mg/l": "mg/L",
    "ug/dl": "ug/dL",
    "mcg/dl": "ug/dL",
    "ng/ml": "ng/mL",
    "pg/ml": "pg/mL",
    "mmol/l": "mmol/L",
    "millimolesperliter": "mmol/L",
    "umol/l": "umol/L",
    "nmol/l": "nmol/L",
    "meq/l": "meq/L",
    "miu/l": "m[IU]/L",
    "iu/l": "[IU]/L",
    "u/l": "U/L",
    "units/l": "U/L",
    "miu/ml": "m[IU]/mL",
    "k/ul": "10*3/uL",
    "10*3/ul": "10*3/uL",
    "m/ul": "10*6/uL",
    "10*6/ul": "10*6/uL",
    "cells/ul": "/uL",
    # pressure
    "mmhg": "mm[Hg]",
    "mm[hg]": "mm[Hg]",
    "millimetersofmercury": "mm[Hg]",
    "cmh2o": "cm[H2O]",
    # rates and time
    "/min": "/min",
    "bpm": "/min",
    "beats/min": "/min",
    "beatsperminute": "/min",
    "breaths/min": "/min",
    "/h": "/h",
    "s": "s",
    "sec": "s",
    "seconds": "s",
    "min": "min",
    "minute": "min",
    "minutes": "min",
    "h": "h",
    "hr": "h",
    "hour": "h",
    "hours": "h",
    "d": "d",
    "day": "d",
    "days": "d",
    "wk": "wk",
    "week": "wk",
    "weeks": "wk",
    "mo": "mo",
    "month": "mo",
    "months": "mo",
    "a": "a",
    "year": "a",
    "years": "a",
    # body metrics and misc
    "kg/m2": "kg/m2",
    "kg/m^2": "kg/m2",
    "ml/min": "mL/min",
    "ml/min/1.73m2": "mL/min/{1.73_m2}",
    "degc": "Cel",
    "c": "Cel",
    "celsius": "Cel",
    "degf": "[degF]",
    "f": "[degF]",
    "fahrenheit": "[degF]",
    "meq": "meq",
    "mmol": "mmol",
    "tablet": "{tbl}",
    "tablets": "{tbl}",
    "capsule": "{capsule}",
    "puff": "{puff}",
    "drop": "[drp]",
    "unit": "[IU]",
    "units": "[IU]",
}

# `\s` already matches U+00A0 and friends for str patterns, so a literal
# non-breaking space here would be redundant and invisible.
_CLEAN = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class UnitResolution:
    """The outcome of looking up a unit string.

    Attributes:
        text: The unit as written, preserved for ``valueQuantity.unit``.
        code: UCUM code, or None when unrecognized.
        system: :data:`UCUM_SYSTEM` when a code was found, else None.
    """

    text: str
    code: str | None = None
    system: str | None = None

    @property
    def is_ucum(self) -> bool:
        """Whether a UCUM code was found."""
        return self.code is not None


def normalize_unit_key(raw: str) -> str:
    """Normalize a unit string for table lookup."""
    cleaned = _CLEAN.sub("", (raw or "").strip().lower())
    # Treat the common superscript/caret spellings as the same key.
    return cleaned.replace("²", "2").replace("³", "3").replace("·", ".")


def resolve_unit(raw: str | None) -> UnitResolution | None:
    """Look up a unit string in the bundled table.

    Returns:
        A :class:`UnitResolution`. When the unit is not in the table, ``code``
        and ``system`` are None and only the original text is carried -- never
        a guessed code. Returns None for an absent or blank input.
    """
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None

    key = normalize_unit_key(text)
    code = UCUM_TABLE.get(key)
    if code is None:
        # A trailing period or plural "s" is common in free text.
        for variant in (key.rstrip("."), key.rstrip("s"), key.rstrip(".s")):
            if variant and variant in UCUM_TABLE:
                code = UCUM_TABLE[variant]
                break

    if code is None:
        return UnitResolution(text=text)
    return UnitResolution(text=text, code=code, system=UCUM_SYSTEM)


def is_known_ucum_code(code: str | None) -> bool:
    """Whether ``code`` is one this table can produce.

    Used by the project validation rule that forbids the UCUM system appearing
    beside a code fhirground did not resolve itself.
    """
    return code is not None and code in set(UCUM_TABLE.values())


def known_unit_count() -> int:
    """Number of distinct UCUM codes in the bundled table."""
    return len(set(UCUM_TABLE.values()))
