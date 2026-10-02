"""Preprocessing: section detection and offset-tracked chunking."""

from fhirground.preprocess.chunking import Chunk, ChunkPlan, chunk_note
from fhirground.preprocess.sections import (
    DEFAULT_SECTION_HEADERS,
    NON_PATIENT_SECTIONS,
    Section,
    detect_sections,
    is_non_patient_section,
    section_at,
)

__all__ = [
    "DEFAULT_SECTION_HEADERS",
    "NON_PATIENT_SECTIONS",
    "Chunk",
    "ChunkPlan",
    "Section",
    "chunk_note",
    "detect_sections",
    "is_non_patient_section",
    "section_at",
]
