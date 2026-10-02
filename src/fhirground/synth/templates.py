"""Rendering synthetic notes, with gold FHIR generated in parallel.

The benchmark design: take structured source records, render a note that states
them in clinical prose, and emit the FHIR those records imply. Because both
sides come from the same records, the gold is exact by construction rather than
annotated -- which is what makes an offline, reproducible benchmark possible.

The templates deliberately inject what makes extraction hard:

* **negated distractors** -- findings the record does *not* contain, stated as
  denials. A pipeline that ignores negation turns these into false positives,
  and there is no way to score that without them.
* **family history** -- conditions attributed to a relative.
* **hedged mentions** -- "possible", "rule out".
* **unit spelling variants** -- ``mg/dL``, ``mg/dl``, ``milligrams per deciliter``.
* **comparators** -- ``<0.01``, ``>140``.
* **abbreviations** -- ``T2DM``, ``HTN``, ``A1c``.
* **light typos** -- a transposition or a doubled letter, which is what fuzzy
  grounding exists to survive.

Every injected distractor is recorded, so a scorer can check that the negated
ones were dropped rather than merely absent.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

#: Conditions available to the renderer: display, code, abbreviations.
CONDITION_BANK: tuple[dict[str, Any], ...] = (
    {
        "display": "Type 2 diabetes mellitus",
        "code": "44054006",
        "abbrev": ("T2DM", "type 2 diabetes"),
    },
    {"display": "Hypertension", "code": "38341003", "abbrev": ("HTN", "hypertension")},
    {"display": "Hyperlipidemia", "code": "55822004", "abbrev": ("hyperlipidemia",)},
    {"display": "Asthma", "code": "195967001", "abbrev": ("asthma",)},
    {
        "display": "Chronic kidney disease",
        "code": "709044004",
        "abbrev": ("CKD", "chronic kidney disease"),
    },
    {"display": "Obesity", "code": "414916001", "abbrev": ("obesity",)},
    {"display": "Hypothyroidism", "code": "40930008", "abbrev": ("hypothyroidism",)},
    {"display": "Depression", "code": "35489007", "abbrev": ("depression",)},
)

#: Medications: display, RxNorm code, dose, unit, route.
MEDICATION_BANK: tuple[dict[str, Any], ...] = (
    {"display": "metformin", "code": "860975", "dose": 500, "unit": "mg", "route": "PO"},
    {"display": "lisinopril", "code": "314076", "dose": 10, "unit": "mg", "route": "PO"},
    {"display": "atorvastatin", "code": "617312", "dose": 20, "unit": "mg", "route": "PO"},
    {"display": "levothyroxine", "code": "966222", "dose": 50, "unit": "mcg", "route": "PO"},
    {"display": "albuterol", "code": "745679", "dose": 2, "unit": "puffs", "route": "inhaled"},
    {"display": "amlodipine", "code": "308136", "dose": 5, "unit": "mg", "route": "PO"},
    {"display": "sertraline", "code": "312940", "dose": 50, "unit": "mg", "route": "PO"},
)

#: Observations: display, LOINC code, value range, unit, category.
OBSERVATION_BANK: tuple[dict[str, Any], ...] = (
    {
        "display": "Hemoglobin A1c",
        "code": "4548-4",
        "low": 5.0,
        "high": 12.0,
        "unit": "%",
        "category": "laboratory",
        "abbrev": ("A1c", "HbA1c", "hemoglobin A1c"),
    },
    {
        "display": "LDL Cholesterol",
        "code": "18262-6",
        "low": 60,
        "high": 190,
        "unit": "mg/dL",
        "category": "laboratory",
        "abbrev": ("LDL", "LDL cholesterol"),
    },
    {
        "display": "Creatinine",
        "code": "2160-0",
        "low": 0.6,
        "high": 2.4,
        "unit": "mg/dL",
        "category": "laboratory",
        "abbrev": ("creatinine",),
    },
    {
        "display": "Potassium",
        "code": "2823-3",
        "low": 3.2,
        "high": 5.4,
        "unit": "mmol/L",
        "category": "laboratory",
        "abbrev": ("potassium",),
    },
    {
        "display": "Heart rate",
        "code": "8867-4",
        "low": 55,
        "high": 110,
        "unit": "bpm",
        "category": "vital-signs",
        "abbrev": ("heart rate", "pulse"),
    },
    {
        "display": "Body Mass Index",
        "code": "39156-5",
        "low": 18.0,
        "high": 41.0,
        "unit": "kg/m2",
        "category": "vital-signs",
        "abbrev": ("BMI",),
    },
    {
        "display": "Body Weight",
        "code": "29463-7",
        "low": 50,
        "high": 130,
        "unit": "kg",
        "category": "vital-signs",
        "abbrev": ("weight",),
    },
)

#: Spelling variants per canonical unit, so the UCUM table is exercised.
UNIT_VARIANTS: dict[str, tuple[str, ...]] = {
    "%": ("%", "percent"),
    "mg/dL": ("mg/dL", "mg/dl", "milligrams per deciliter"),
    "mmol/L": ("mmol/L", "mmol/l"),
    "bpm": ("bpm", "beats/min"),
    "kg/m2": ("kg/m2", "kg/m^2"),
    "kg": ("kg", "kilograms"),
    "mg": ("mg", "milligrams"),
    "mcg": ("mcg", "micrograms"),
    "puffs": ("puffs",),
}

SNOMED = "http://snomed.info/sct"
RXNORM = "http://www.nlm.nih.gov/research/umls/rxnorm"
LOINC = "http://loinc.org"


@dataclass
class SourceRecord:
    """The structured truth a note is rendered from."""

    patient_id: str
    note_datetime: datetime
    conditions: list[dict[str, Any]] = field(default_factory=list)
    medications: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)

    @property
    def patient_ref(self) -> str:
        """FHIR reference for the patient."""
        return f"Patient/{self.patient_id}"


@dataclass
class RenderedNote:
    """A synthetic note plus the gold it was rendered from.

    Attributes:
        note_id: Identifier.
        text: The note text.
        gold: Gold FHIR resources, generated from the same source records.
        distractors: Injected findings that must **not** be emitted, with the
            reason each should be dropped. Lets a scorer verify that negation
            and family history were handled rather than that they happened to
            be absent.
    """

    note_id: str
    patient_ref: str
    note_datetime: datetime
    text: str
    gold: list[dict[str, Any]] = field(default_factory=list)
    distractors: list[dict[str, str]] = field(default_factory=list)


def _typo(word: str, rng: random.Random) -> str:
    """Introduce one light, recoverable typo."""
    if len(word) < 5:
        return word
    i = rng.randrange(1, len(word) - 2)
    if rng.random() < 0.5:
        return word[:i] + word[i + 1] + word[i] + word[i + 2 :]  # transposition
    return word[:i] + word[i] + word[i:]  # doubled letter


def _unit_variant(unit: str, rng: random.Random) -> str:
    return rng.choice(UNIT_VARIANTS.get(unit, (unit,)))


def _value_for(obs: dict[str, Any], rng: random.Random) -> float:
    low, high = obs["low"], obs["high"]
    if isinstance(low, int) and isinstance(high, int):
        return float(rng.randint(low, high))
    return round(rng.uniform(float(low), float(high)), 1)


def make_source_record(patient_index: int, rng: random.Random, *, base: datetime) -> SourceRecord:
    """Build one patient's structured record.

    Used when no Synthea output is available. The shape matches what
    :mod:`fhirground.synth.notes_from_synthea` extracts from a real Synthea
    bundle, so the renderer does not care which produced it.
    """
    patient_id = f"synth-{patient_index:05d}"
    when = base + timedelta(days=rng.randint(0, 900), hours=rng.randint(8, 17))

    conditions = [dict(c) for c in rng.sample(CONDITION_BANK, k=rng.randint(1, 3))]
    medications = [dict(m) for m in rng.sample(MEDICATION_BANK, k=rng.randint(1, 3))]
    observations = []
    for obs in rng.sample(OBSERVATION_BANK, k=rng.randint(1, 4)):
        entry = dict(obs)
        entry["value"] = _value_for(obs, rng)
        observations.append(entry)

    return SourceRecord(
        patient_id=patient_id,
        note_datetime=when,
        conditions=conditions,
        medications=medications,
        observations=observations,
    )


def _gold_condition(record: SourceRecord, condition: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "resourceType": "Condition",
        "id": f"gold-{record.patient_id}-cond-{index}",
        "subject": {"reference": record.patient_ref},
        "verificationStatus": {
            "coding": [
                {
                    "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                    "code": "confirmed",
                }
            ]
        },
        "code": {
            "coding": [
                {"system": SNOMED, "code": condition["code"], "display": condition["display"]}
            ],
            "text": condition["display"],
        },
    }


def _gold_medication(
    record: SourceRecord, medication: dict[str, Any], index: int
) -> dict[str, Any]:
    return {
        "resourceType": "MedicationRequest",
        "id": f"gold-{record.patient_id}-med-{index}",
        "status": "active",
        "intent": "order",
        "subject": {"reference": record.patient_ref},
        "medicationCodeableConcept": {
            "coding": [
                {"system": RXNORM, "code": medication["code"], "display": medication["display"]}
            ],
            "text": medication["display"],
        },
    }


def _gold_observation(
    record: SourceRecord, observation: dict[str, Any], index: int
) -> dict[str, Any]:
    return {
        "resourceType": "Observation",
        "id": f"gold-{record.patient_id}-obs-{index}",
        "status": "final",
        "subject": {"reference": record.patient_ref},
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code": observation["category"],
                    }
                ]
            }
        ],
        "code": {
            "coding": [
                {"system": LOINC, "code": observation["code"], "display": observation["display"]}
            ],
            "text": observation["display"],
        },
        "valueQuantity": {"value": observation["value"], "unit": observation["unit"]},
    }


def render_note(
    record: SourceRecord,
    note_index: int,
    rng: random.Random,
    *,
    inject_distractors: bool = True,
) -> RenderedNote:
    """Render one note and its gold from a source record."""
    note_id = f"synth-note-{note_index:05d}"
    lines: list[str] = []
    gold: list[dict[str, Any]] = []
    distractors: list[dict[str, str]] = []

    lines.append("Chief Complaint: routine follow-up.")
    lines.append("")
    lines.append("History of Present Illness:")
    lines.append(
        f"Patient seen today, {record.note_datetime.strftime('%B %-d, %Y')}, for scheduled review."
        if hasattr(record.note_datetime, "strftime")
        else "Patient seen today for scheduled review."
    )

    # --- assessment: the real conditions ------------------------------
    lines.append("")
    lines.append("Assessment:")
    for i, condition in enumerate(record.conditions):
        name = rng.choice(condition["abbrev"])
        if rng.random() < 0.18:
            name = _typo(name, rng)
        lines.append(f"{name.capitalize()}, stable on current therapy.")
        gold.append(_gold_condition(record, condition, i))

    if inject_distractors:
        # Hedged mention: in the record's own terms but not asserted. The
        # pipeline should emit it as provisional, so it is not a distractor to
        # drop -- it is recorded so a scorer knows why a provisional appeared.
        absent = [
            c for c in CONDITION_BANK if c["code"] not in {x["code"] for x in record.conditions}
        ]
        if absent and rng.random() < 0.5:
            hedged = rng.choice(absent)
            lines.append(f"Possible {rng.choice(hedged['abbrev'])}; will reassess.")
            distractors.append(
                {"text": hedged["display"], "expect": "provisional", "kind": "hedged"}
            )

        # Negated distractor: must not be emitted at all.
        if absent:
            negated = rng.choice(absent)
            lines.append(f"Denies {rng.choice(negated['abbrev'])}.")
            distractors.append({"text": negated["display"], "expect": "dropped", "kind": "negated"})

    # --- observations --------------------------------------------------
    if record.observations:
        lines.append("")
        lines.append("Laboratory:")
        for i, observation in enumerate(record.observations):
            label = rng.choice(observation["abbrev"])
            unit = _unit_variant(observation["unit"], rng)
            value = observation["value"]
            shown = f"{value:g}"
            if rng.random() < 0.15:
                # A comparator. The gold keeps the number, so a pipeline that
                # drops the comparator still matches on value -- the comparator
                # is tested directly in the unit tests instead.
                shown = f"{rng.choice(['<', '>', '<=', '>='])}{shown}"
            lines.append(f"{label} {shown} {unit} today.")
            gold.append(_gold_observation(record, observation, i))

    # --- medications ---------------------------------------------------
    if record.medications:
        lines.append("")
        lines.append("Plan:")
        for i, medication in enumerate(record.medications):
            unit = _unit_variant(medication["unit"], rng)
            verb = rng.choice(("Start", "Continue", "Increase"))
            freq = rng.choice(("BID", "daily", "QHS", "TID"))
            lines.append(
                f"{verb} {medication['display']} {medication['dose']:g} {unit} "
                f"{medication['route']} {freq}."
            )
            gold.append(_gold_medication(record, medication, i))

    if inject_distractors:
        lines.append("")
        lines.append("Family History:")
        family = rng.choice(CONDITION_BANK)
        relative = rng.choice(("Mother", "Father", "Sister", "Brother"))
        lines.append(f"{relative} has {rng.choice(family['abbrev'])}.")
        distractors.append({"text": family["display"], "expect": "dropped", "kind": "family"})

    return RenderedNote(
        note_id=note_id,
        patient_ref=record.patient_ref,
        note_datetime=record.note_datetime,
        text="\n".join(lines) + "\n",
        gold=gold,
        distractors=distractors,
    )


def value_sets_from_records(records: list[SourceRecord]) -> dict[str, list[dict[str, Any]]]:
    """Build value sets covering exactly the codes the records use.

    Derived from the fixtures rather than shipped as terminology: the docs are
    explicit that a real deployment supplies licensed value sets, and these
    exist only so the tests and the demo can resolve codes at all.
    """
    conditions: dict[str, dict[str, Any]] = {}
    medications: dict[str, dict[str, Any]] = {}
    observations: dict[str, dict[str, Any]] = {}

    for record in records:
        for condition in record.conditions:
            conditions[condition["code"]] = {
                "system": SNOMED,
                "code": condition["code"],
                "display": condition["display"],
                "synonyms": "|".join(condition.get("abbrev") or ()),
            }
        for medication in record.medications:
            medications[medication["code"]] = {
                "system": RXNORM,
                "code": medication["code"],
                "display": medication["display"],
                "synonyms": medication["display"],
            }
        for observation in record.observations:
            observations[observation["code"]] = {
                "system": LOINC,
                "code": observation["code"],
                "display": observation["display"],
                "synonyms": "|".join(observation.get("abbrev") or ()),
            }

    return {
        "conditions": list(conditions.values()),
        "medications": list(medications.values()),
        "observations": list(observations.values()),
    }
