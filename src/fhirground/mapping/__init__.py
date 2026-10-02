"""Mention to FHIR R4B mapping.

Every builder here takes a :class:`~fhirground.terminology.base.Resolution`
rather than a code string. That is not a style preference -- it is how SPEC
§4.5's rule that "every coding came from a Resolution object" is enforced
structurally instead of by review.
"""

from fhirground.mapping.condition import (
    CLINICAL_STATUS_SYSTEM,
    CON4_ALLOWED_CLINICAL_STATUS,
    VERIFICATION_STATUS_SYSTEM,
    build_condition,
)
from fhirground.mapping.extensions import (
    CALIBRATED_SUB,
    DEFAULT_EXTENSION_BASE,
    ExtensionUrls,
    confidence_extension,
    effective_inferred_extension,
    evidence_extension,
    read_calibrated,
    read_confidence,
    read_evidence,
)
from fhirground.mapping.medication_request import build_medication_request
from fhirground.mapping.observation import CATEGORY_SYSTEM, build_observation
from fhirground.mapping.provenance import build_bundle, build_provenance

__all__ = [
    "CALIBRATED_SUB",
    "CATEGORY_SYSTEM",
    "CLINICAL_STATUS_SYSTEM",
    "CON4_ALLOWED_CLINICAL_STATUS",
    "DEFAULT_EXTENSION_BASE",
    "VERIFICATION_STATUS_SYSTEM",
    "ExtensionUrls",
    "build_bundle",
    "build_condition",
    "build_medication_request",
    "build_observation",
    "build_provenance",
    "confidence_extension",
    "effective_inferred_extension",
    "evidence_extension",
    "read_calibrated",
    "read_confidence",
    "read_evidence",
]
