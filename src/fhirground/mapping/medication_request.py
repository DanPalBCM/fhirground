"""Mention to FHIR R4B ``MedicationRequest``.

R4B keeps the R4 choice element ``medicationCodeableConcept``. R5 replaced it
with a single ``medication`` of type ``CodeableReference``, which is why
fhirground pins the R4B models rather than taking the library's default (see
``docs/decisions.md`` D-002).

``authoredOn`` is set **only** from an explicitly resolved date, never inferred
from the note timestamp. ``authoredOn`` is the date an order was written, and
defaulting it to "when the note was typed" would fabricate a clinical fact that
looks authoritative. An Observation's time can be inferred because a flag
records that it was; an order date cannot.
"""

from __future__ import annotations

from typing import Any

from fhirground.assertion import EmissionDecision, RequestStatus
from fhirground.exceptions import MappingError
from fhirground.grounding import Grounding
from fhirground.io import NoteInput
from fhirground.mapping.extensions import (
    ExtensionUrls,
    confidence_extension,
    evidence_extension,
)
from fhirground.normalize.dates import ResolvedDate
from fhirground.normalize.units import resolve_unit
from fhirground.normalize.values import parse_dose
from fhirground.schemas.mention import Mention
from fhirground.terminology.base import Resolution

#: v1 emits orders, so intent is always ``order``. A ``plan`` or ``proposal``
#: would need evidence of intent the note rarely carries explicitly.
DEFAULT_INTENT = "order"

_STATUS_VALUES: dict[RequestStatus, str] = {
    RequestStatus.ACTIVE: "active",
    RequestStatus.STOPPED: "stopped",
}


def build_medication_request(
    mention: Mention,
    resolution: Resolution,
    grounding: Grounding,
    decision: EmissionDecision,
    note: NoteInput,
    *,
    confidence: float,
    calibrated: bool = False,
    resource_id: str,
    authored_on: ResolvedDate | None = None,
    include_quotes: bool = True,
    urls: ExtensionUrls | None = None,
) -> dict[str, Any]:
    """Build a ``MedicationRequest`` as a plain dict.

    Args:
        mention: The source mention.
        resolution: The resolved medication concept. A
            :class:`~fhirground.terminology.base.Resolution`, never a string.
        grounding: Where the evidence sits in the note.
        decision: Policy verdict, supplying ``status``.
        note: Source note, for ``subject`` and ``encounter``.
        confidence: Score for the confidence extension.
        calibrated: Whether a fitted calibrator produced the score.
        resource_id: Deterministic id from the pipeline.
        authored_on: An explicitly resolved date. Ignored when it is marked
            ``inferred``, because ``authoredOn`` must not be guessed.
        include_quotes: Carry the evidence quote.
        urls: Configured extension URLs.

    Returns:
        A ``MedicationRequest`` resource dict.

    Raises:
        MappingError: If the policy did not authorize emission.
    """
    if not decision.emit:
        raise MappingError(
            "build_medication_request called for a dropped mention "
            f"({decision.reason.value if decision.reason else 'unknown'})"
        )
    if decision.request_status is None:
        raise MappingError("a MedicationRequest needs a status from the policy")

    u = urls or ExtensionUrls()
    resource: dict[str, Any] = {
        "resourceType": "MedicationRequest",
        "id": resource_id,
        "status": _STATUS_VALUES[decision.request_status],
        "intent": DEFAULT_INTENT,
        "subject": {"reference": note.patient_ref},
        "medicationCodeableConcept": resolution.as_codeable_concept(),
    }

    if note.encounter_ref:
        resource["encounter"] = {"reference": note.encounter_ref}

    # Only an explicit date. `inferred` dates are rejected here on purpose.
    if authored_on is not None and not authored_on.inferred:
        resource["authoredOn"] = authored_on.to_fhir()

    dosage = _build_dosage(mention, grounding)
    if dosage:
        resource["dosageInstruction"] = [dosage]

    resource["extension"] = [
        confidence_extension(confidence, calibrated=calibrated, urls=u),
        evidence_extension(
            grounding,
            source_reference=note.source_reference,
            include_quote=include_quotes,
            urls=u,
        ),
    ]
    return resource


def _build_dosage(mention: Mention, grounding: Grounding) -> dict[str, Any] | None:
    """Build ``dosageInstruction[0]`` from whatever the mention supports.

    ``text`` is the evidence sentence, which is the most faithful thing to put
    there: it is what the prescriber actually wrote, verbatim and grounded.
    Structured dose and route are added only when they parse cleanly.
    """
    dosage: dict[str, Any] = {}

    text = grounding.quote.strip() or mention.evidence.strip()
    if text:
        dosage["text"] = text

    dose = parse_dose(mention.dose_value, mention.dose_unit)
    if dose is not None:
        value, unit_text = dose
        quantity: dict[str, Any] = {"value": value, "unit": unit_text}
        unit = resolve_unit(unit_text)
        if unit is not None and unit.is_ucum:
            quantity["system"] = unit.system
            quantity["code"] = unit.code
        dosage["doseAndRate"] = [{"doseQuantity": quantity}]

    if mention.route:
        dosage["route"] = {"text": mention.route}

    # `frequency_text` has no faithful structured home: FHIR `timing.repeat`
    # wants a count and a period, and "BID PRN" does not decompose reliably.
    # It is appended to the human-readable text instead of being parsed into a
    # Timing that would assert more precision than the note supports.
    if mention.frequency_text and dosage.get("text"):
        if mention.frequency_text.lower() not in dosage["text"].lower():
            dosage["text"] = f"{dosage['text']} ({mention.frequency_text})"
    elif mention.frequency_text:
        dosage["text"] = mention.frequency_text

    return dosage or None


def add_evidence(
    resource: dict[str, Any],
    grounding: Grounding,
    note: NoteInput,
    *,
    include_quotes: bool = True,
    urls: ExtensionUrls | None = None,
) -> None:
    """Append another evidence extension to a merged resource."""
    u = urls or ExtensionUrls()
    resource.setdefault("extension", []).append(
        evidence_extension(
            grounding,
            source_reference=note.source_reference,
            include_quote=include_quotes,
            urls=u,
        )
    )
