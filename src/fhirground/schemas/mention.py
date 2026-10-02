"""The mention schema: what the LLM is allowed to say.

This is the narrow waist of the whole design. The model produces *mentions* --
plain-language findings with a verbatim evidence quote -- and deterministic code
produces FHIR. FHIR structure never depends on the model's formatting.

**There are no code fields here, by design.** A model cannot emit a SNOMED,
LOINC, or RxNorm code through this schema because the schema has nowhere to put
one. Codes enter only through a
:class:`~fhirground.terminology.base.TerminologyResolver`, and the mapping
functions accept only :class:`~fhirground.terminology.base.Resolution` objects,
never raw strings. A property test asserts the invariant over adversarial input.

The JSON Schema generated from these models is the constrained-decoding target
for both backends, so it stays inside the subset Bedrock tool ``inputSchema``
and vLLM guided decoding both accept: flat objects, enums, nullable fields via
type unions, no ``oneOf``/``anyOf`` on objects, and no ``$ref`` recursion.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: Maximum length of a mention's surface text, enforced in the schema so a
#: model cannot return a paragraph where a concept was expected.
MAX_TEXT_LEN = 200
#: Maximum length of an evidence quote. Long enough for a clinical sentence.
MAX_EVIDENCE_LEN = 300


class MentionType(StrEnum):
    """The three things v1 extracts."""

    CONDITION = "condition"
    MEDICATION = "medication"
    OBSERVATION = "observation"


class Assertion(StrEnum):
    """How the note asserts the finding.

    This is the field that keeps negated and family-history mentions from
    becoming affirmed facts. The emission policy per value is in
    :mod:`fhirground.assertion`.
    """

    PRESENT = "present"
    NEGATED = "negated"
    POSSIBLE = "possible"
    HISTORICAL = "historical"
    FAMILY = "family"
    HYPOTHETICAL = "hypothetical"


class ClinicalStatusHint(StrEnum):
    """An optional hint at FHIR ``Condition.clinicalStatus``.

    A hint, not an assignment: the mapper decides whether to use it, and omits
    ``clinicalStatus`` entirely when there is no hint rather than guessing
    ``active``.
    """

    ACTIVE = "active"
    RESOLVED = "resolved"
    INACTIVE = "inactive"
    REMISSION = "remission"


class MedAction(StrEnum):
    """What the note says is being done with a medication."""

    START = "start"
    CONTINUE = "continue"
    CHANGE = "change"
    STOP = "stop"
    MENTION = "mention"


class ObsCategory(StrEnum):
    """FHIR observation category, restricted to the two v1 supports."""

    VITAL_SIGNS = "vital-signs"
    LABORATORY = "laboratory"


class ValueComparator(StrEnum):
    """FHIR ``Quantity.comparator`` values."""

    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="


class Mention(BaseModel):
    """One finding the model claims the note contains.

    Only four fields are required. Everything else is optional because a model
    that must fill every field will invent values for the ones it cannot
    support -- and an invented dose is worse than a missing one.
    """

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    mention_type: MentionType
    text: str = Field(
        min_length=1,
        max_length=MAX_TEXT_LEN,
        description="The concept as the note expresses it, in plain language.",
    )
    evidence: str = Field(
        min_length=1,
        max_length=MAX_EVIDENCE_LEN,
        description="Verbatim quote copied from the note. Must appear in the note text.",
    )
    assertion: Assertion

    # -- condition ------------------------------------------------------
    clinical_status_hint: ClinicalStatusHint | None = None
    onset_text: str | None = Field(
        default=None,
        max_length=MAX_TEXT_LEN,
        description="Onset as the note expresses it, e.g. 'since 2019'.",
    )

    # -- medication -----------------------------------------------------
    med_action: MedAction | None = None
    dose_value: float | None = None
    dose_unit: str | None = Field(default=None, max_length=50)
    route: str | None = Field(default=None, max_length=50)
    frequency_text: str | None = Field(default=None, max_length=100)

    # -- observation ----------------------------------------------------
    obs_category: ObsCategory | None = None
    value_number: float | None = None
    value_comparator: ValueComparator | None = None
    value_text: str | None = Field(default=None, max_length=MAX_TEXT_LEN)
    value_unit: str | None = Field(default=None, max_length=50)
    date_text: str | None = Field(
        default=None,
        max_length=MAX_TEXT_LEN,
        description="Date or relative phrase as written, e.g. '3 days ago'.",
    )

    @field_validator("text", "evidence", mode="before")
    @classmethod
    def _strip(cls, v: object) -> object:
        return v.strip() if isinstance(v, str) else v

    @field_validator(
        "onset_text",
        "dose_unit",
        "route",
        "frequency_text",
        "value_text",
        "value_unit",
        "date_text",
        mode="before",
    )
    @classmethod
    def _blank_to_none(cls, v: object) -> object:
        """Treat an empty or whitespace-only optional string as absent.

        Models frequently emit ``""`` instead of ``null`` for a field they have
        nothing to say about. Carrying that through would produce a FHIR
        resource with an empty ``route.text``.
        """
        if isinstance(v, str):
            s = v.strip()
            return s if s else None
        return v

    @property
    def normalized_text(self) -> str:
        """Lower-cased, whitespace-collapsed surface form, for deduplication."""
        return " ".join(self.text.lower().split())


class MentionList(BaseModel):
    """The complete model response: a list of mentions and nothing else."""

    model_config = ConfigDict(extra="forbid")

    mentions: list[Mention] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.mentions)

    def __iter__(self) -> Any:  # type: ignore[override]
        return iter(self.mentions)

    def by_type(self, mention_type: MentionType) -> list[Mention]:
        """Mentions of one type, in order."""
        return [m for m in self.mentions if m.mention_type is mention_type]


#: Field names that may appear on a mention. Used by the no-invented-codes
#: guard: if a future edit adds a code-shaped field, that test fails.
MENTION_FIELDS: frozenset[str] = frozenset(Mention.model_fields)

#: Substrings that must never appear in a mention field name. The hard rule in
#: SPEC §0 is that the LLM never emits medical codes; the cheapest way to keep
#: it true is to make a code field impossible to add by accident.
FORBIDDEN_FIELD_SUBSTRINGS: tuple[str, ...] = (
    "code",
    "coding",
    "snomed",
    "loinc",
    "rxnorm",
    "icd",
    "concept_id",
    "system",
)


def assert_no_code_fields() -> None:
    """Raise if the mention schema has acquired a code-bearing field.

    Called by a test rather than at import time, so the check is visible in the
    suite rather than being a silent side effect.
    """
    offenders = {
        name for name in MENTION_FIELDS for bad in FORBIDDEN_FIELD_SUBSTRINGS if bad in name.lower()
    }
    if offenders:
        raise AssertionError(
            "the mention schema must contain no code fields (SPEC §0 hard rule), "
            f"but found: {sorted(offenders)}. Codes may only enter through a "
            "TerminologyResolver."
        )
