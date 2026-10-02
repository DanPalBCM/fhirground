"""Shared fixtures. Every test uses synthetic data and makes no network calls."""

from __future__ import annotations

import os
from datetime import datetime

import pytest

from fhirground.io import NoteInput
from fhirground.pipeline import PipelineConfig
from fhirground.schemas.mention import (
    Assertion,
    MedAction,
    Mention,
    MentionType,
    ObsCategory,
)
from fhirground.terminology import Chain, LocalValueSet, ResolverConfig

#: Env vars that opt a live-backend test in.
LIVE_MARKERS = {
    "live_bedrock": "FHIRGROUND_LIVE_BEDROCK",
    "live_vllm": "FHIRGROUND_LIVE_VLLM",
    "live_rxnav": "FHIRGROUND_LIVE_RXNAV",
}

NOTE_DT = datetime(2026, 3, 14, 10, 30)

#: The note from SPEC §5.2, used by the definition-of-done test.
SPEC_NOTE_TEXT = (
    "Assessment: Type 2 diabetes, poorly controlled. A1c 9.2% today. "
    "Start metformin 500 mg PO BID. Denies chest pain. Mother has hypertension."
)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip live-backend tests unless their env var is set."""
    for item in items:
        for marker, env in LIVE_MARKERS.items():
            if marker in item.keywords and not os.environ.get(env):
                item.add_marker(pytest.mark.skip(reason=f"needs {env} set (network + credentials)"))


# ---------------------------------------------------------------------------
# notes
# ---------------------------------------------------------------------------


@pytest.fixture
def spec_note() -> NoteInput:
    """The SPEC §5.2 example note."""
    return NoteInput(
        note_id="n1",
        patient_ref="Patient/123",
        note_datetime=NOTE_DT,
        text=SPEC_NOTE_TEXT,
        document_reference_id="docref-n1",
    )


def make_note(text: str, note_id: str = "n1", **kwargs: object) -> NoteInput:
    """Build a note with terse arguments."""
    defaults: dict[str, object] = {
        "note_id": note_id,
        "patient_ref": "Patient/123",
        "note_datetime": NOTE_DT,
        "text": text,
    }
    defaults.update(kwargs)
    return NoteInput.model_validate(defaults)


# ---------------------------------------------------------------------------
# mentions
# ---------------------------------------------------------------------------


def make_mention(
    mention_type: MentionType = MentionType.CONDITION,
    text: str = "type 2 diabetes",
    evidence: str = "Type 2 diabetes, poorly controlled",
    assertion: Assertion = Assertion.PRESENT,
    **kwargs: object,
) -> Mention:
    """Build a mention with terse arguments."""
    return Mention(
        mention_type=mention_type,
        text=text,
        evidence=evidence,
        assertion=assertion,
        **kwargs,  # type: ignore[arg-type]
    )


@pytest.fixture
def condition_mention() -> Mention:
    return make_mention()


@pytest.fixture
def medication_mention() -> Mention:
    return make_mention(
        MentionType.MEDICATION,
        text="metformin",
        evidence="Start metformin 500 mg PO BID",
        med_action=MedAction.START,
        dose_value=500,
        dose_unit="mg",
        route="PO",
        frequency_text="BID",
    )


@pytest.fixture
def observation_mention() -> Mention:
    return make_mention(
        MentionType.OBSERVATION,
        text="a1c",
        evidence="A1c 9.2% today",
        obs_category=ObsCategory.LABORATORY,
        value_number=9.2,
        value_unit="%",
        date_text="today",
    )


# ---------------------------------------------------------------------------
# terminology
# ---------------------------------------------------------------------------

SNOMED = "http://snomed.info/sct"
RXNORM = "http://www.nlm.nih.gov/research/umls/rxnorm"
LOINC = "http://loinc.org"

CONDITION_ROWS = [
    {
        "system": SNOMED,
        "code": "44054006",
        "display": "Type 2 diabetes mellitus",
        "synonyms": "type 2 diabetes|T2DM|DM2",
    },
    {
        "system": SNOMED,
        "code": "38341003",
        "display": "Hypertension",
        "synonyms": "HTN|hypertension|high blood pressure",
    },
    {"system": SNOMED, "code": "29857009", "display": "Chest pain", "synonyms": "chest pain"},
    {"system": SNOMED, "code": "233604007", "display": "Pneumonia", "synonyms": "pneumonia"},
]
MEDICATION_ROWS = [
    {
        "system": RXNORM,
        "code": "860975",
        "display": "metformin hydrochloride 500 MG Oral Tablet",
        "synonyms": "metformin|metformin 500|glucophage 500",
    },
    {
        "system": RXNORM,
        "code": "314076",
        "display": "lisinopril 10 MG Oral Tablet",
        "synonyms": "lisinopril",
    },
]
OBSERVATION_ROWS = [
    {
        "system": LOINC,
        "code": "4548-4",
        "display": "Hemoglobin A1c",
        "synonyms": "a1c|hba1c|hemoglobin a1c",
    },
    {"system": LOINC, "code": "8867-4", "display": "Heart rate", "synonyms": "heart rate|pulse"},
]


@pytest.fixture
def resolvers() -> ResolverConfig:
    """A resolver config covering the demo concepts."""
    return ResolverConfig(
        condition=Chain([LocalValueSet.from_entries(CONDITION_ROWS, name="conds")]),
        medication=Chain([LocalValueSet.from_entries(MEDICATION_ROWS, name="meds")]),
        observation=Chain([LocalValueSet.from_entries(OBSERVATION_ROWS, name="obs")]),
    )


@pytest.fixture
def known_codes(resolvers: ResolverConfig) -> set[tuple[str, str]]:
    """Every code the fixture resolvers could emit."""
    return resolvers.all_known_codes()


@pytest.fixture
def fast_config() -> PipelineConfig:
    """A pipeline config with one sample and no cache."""
    return PipelineConfig(samples=1, max_repairs=2)
