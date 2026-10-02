"""Synthetic notes with exact gold FHIR. The only data fhirground ships."""

from fhirground.synth.notes_from_synthea import (
    EPOCH,
    SYNTHEA_VERSION,
    build_benchmark,
    generate_records,
    records_from_synthea,
    write_benchmark,
)
from fhirground.synth.templates import (
    CONDITION_BANK,
    LOINC,
    MEDICATION_BANK,
    OBSERVATION_BANK,
    RXNORM,
    SNOMED,
    RenderedNote,
    SourceRecord,
    make_source_record,
    render_note,
    value_sets_from_records,
)

__all__ = [
    "CONDITION_BANK",
    "EPOCH",
    "LOINC",
    "MEDICATION_BANK",
    "OBSERVATION_BANK",
    "RXNORM",
    "SNOMED",
    "SYNTHEA_VERSION",
    "RenderedNote",
    "SourceRecord",
    "build_benchmark",
    "generate_records",
    "make_source_record",
    "records_from_synthea",
    "render_note",
    "value_sets_from_records",
    "write_benchmark",
]
