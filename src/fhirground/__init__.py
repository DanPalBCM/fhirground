"""fhirground: grounded, validated FHIR R4 extraction from clinical notes.

The design in one sentence: **the LLM produces mentions, deterministic code
produces FHIR.** Asking a model to emit FHIR directly yields invalid resources,
misplaced fields, and hallucinated codes. Splitting the problem means FHIR
structure never depends on model formatting.

Four guarantees, each enforced in code rather than hoped for:

* **Grounded.** Every emitted resource carries evidence offsets that index the
  original note. A mention whose quote cannot be found is dropped and counted.
* **No invented codes.** The mention schema has no code fields, and every
  mapping function accepts a ``Resolution`` rather than a string -- so a code
  cannot reach a resource without having come from a terminology resolver. A
  property test drives adversarial input through the pipeline to check it.
* **Validated.** Every resource parses under the pinned FHIR R4B models and
  passes the project rules, or it is dropped with a reason.
* **Assertion-aware.** Negated, family-history, and hypothetical mentions are
  never emitted as affirmed facts.

Quickstart::

    import fhirground as fg

    pipe = fg.Pipeline(
        fg.PipelineConfig(),
        backend=fg.backends.MockBackend(),
        resolvers=fg.terminology.ResolverConfig(),
    )
    result = pipe.run_note(
        fg.NoteInput(
            note_id="n1",
            patient_ref="Patient/123",
            note_datetime="2026-03-14T10:30:00Z",
            text="Assessment: Type 2 diabetes. Start metformin 500 mg PO BID.",
        )
    )
    print(result.bundle)
    for mention, reason in result.dropped:
        print(reason.value, mention.text)
"""

from __future__ import annotations

__version__ = "1.0.0"

from fhirground import backends, mapping, normalize, preprocess, synth, terminology
from fhirground.assertion import (
    DropReason,
    EmissionDecision,
    RequestStatus,
    VerificationStatus,
    decide,
)
from fhirground.confidence import (
    DEFAULT_WEIGHTS,
    Calibrator,
    ConfidenceComponents,
    aggregate,
    self_consistency,
    validation_component,
)
from fhirground.dedup import MentionGroup, dedup_key, dedup_mentions
from fhirground.evaluate import MatchSpec, ScoreReport, score_resources
from fhirground.exceptions import (
    BackendError,
    CalibrationError,
    ConfigError,
    FhirgroundError,
    GroundingError,
    InputError,
    MappingError,
    SchemaValidationError,
    TerminologyError,
    ValidationRuleError,
)
from fhirground.export import write_outputs
from fhirground.extract import CallCache, extract_chunk, parse_mentions
from fhirground.grounding import Grounding, GroundingStrategy, ground_mention
from fhirground.io import NoteInput, read_notes, read_notes_fhir, read_notes_jsonl
from fhirground.pipeline import (
    MentionTrace,
    NoteResult,
    Pipeline,
    PipelineConfig,
    run_notes,
    summarize,
)
from fhirground.schemas import (
    Assertion,
    ClinicalStatusHint,
    MedAction,
    Mention,
    MentionList,
    MentionType,
    ObsCategory,
    ValueComparator,
    build_mention_schema,
    load_mention_schema,
)
from fhirground.validate import (
    ValidationReport,
    run_hl7_validator,
    validate_bundle,
    validate_resource,
)

__all__ = [
    "DEFAULT_WEIGHTS",
    "Assertion",
    "BackendError",
    "CalibrationError",
    "Calibrator",
    "CallCache",
    "ClinicalStatusHint",
    "ConfidenceComponents",
    "ConfigError",
    "DropReason",
    "EmissionDecision",
    "FhirgroundError",
    "Grounding",
    "GroundingError",
    "GroundingStrategy",
    "InputError",
    "MappingError",
    "MatchSpec",
    "MedAction",
    "Mention",
    "MentionGroup",
    "MentionList",
    "MentionTrace",
    "MentionType",
    "NoteInput",
    "NoteResult",
    "ObsCategory",
    "Pipeline",
    "PipelineConfig",
    "RequestStatus",
    "SchemaValidationError",
    "ScoreReport",
    "TerminologyError",
    "ValidationReport",
    "ValidationRuleError",
    "ValueComparator",
    "VerificationStatus",
    "__version__",
    "aggregate",
    "backends",
    "build_mention_schema",
    "decide",
    "dedup_key",
    "dedup_mentions",
    "extract_chunk",
    "ground_mention",
    "load_mention_schema",
    "mapping",
    "normalize",
    "parse_mentions",
    "preprocess",
    "read_notes",
    "read_notes_fhir",
    "read_notes_jsonl",
    "run_hl7_validator",
    "run_notes",
    "score_resources",
    "self_consistency",
    "summarize",
    "synth",
    "terminology",
    "validate_bundle",
    "validate_resource",
    "validation_component",
    "write_outputs",
]
