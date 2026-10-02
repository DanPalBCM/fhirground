"""Mention to FHIR R4B ``Observation``.

Two FHIR invariants apply, quoted verbatim from the R4 StructureDefinition:

* **obs-6** (*error*): "dataAbsentReason SHALL only be present if
  Observation.value[x] is not present". fhirground never sets
  ``dataAbsentReason``, so this holds; the project rule checks it.
* **obs-7** (*error*): "If Observation.code is the same as an
  Observation.component.code then the value element associated with the code
  SHALL NOT be present". v1 emits no ``component``, so this holds too.

The value rule that matters in practice: a number becomes ``valueQuantity``
only when a number was actually parsed. Everything else becomes
``valueString``, and a blood pressure pair like ``120/80`` stays a string
rather than being split into invented systolic and diastolic components.
"""

from __future__ import annotations

from typing import Any

from fhirground.assertion import EmissionDecision
from fhirground.exceptions import MappingError
from fhirground.grounding import Grounding
from fhirground.io import NoteInput
from fhirground.mapping.extensions import (
    ExtensionUrls,
    confidence_extension,
    effective_inferred_extension,
    evidence_extension,
)
from fhirground.normalize.dates import ResolvedDate
from fhirground.normalize.units import resolve_unit
from fhirground.normalize.values import ParsedValue
from fhirground.schemas.mention import Mention, ObsCategory
from fhirground.terminology.base import Resolution

#: Terminology system for ``Observation.category``.
CATEGORY_SYSTEM = "http://terminology.hl7.org/CodeSystem/observation-category"

#: v1 emits only final results. A `preliminary` status would need evidence the
#: note rarely carries.
DEFAULT_STATUS = "final"

_CATEGORY_DISPLAY: dict[ObsCategory, str] = {
    ObsCategory.VITAL_SIGNS: "Vital Signs",
    ObsCategory.LABORATORY: "Laboratory",
}


def build_observation(
    mention: Mention,
    resolution: Resolution,
    grounding: Grounding,
    decision: EmissionDecision,
    note: NoteInput,
    *,
    confidence: float,
    calibrated: bool = False,
    resource_id: str,
    value: ParsedValue | None = None,
    effective: ResolvedDate | None = None,
    include_quotes: bool = True,
    mark_inferred_effective: bool = True,
    urls: ExtensionUrls | None = None,
) -> dict[str, Any]:
    """Build an ``Observation`` as a plain dict.

    Args:
        mention: The source mention.
        resolution: The resolved concept, as a
            :class:`~fhirground.terminology.base.Resolution`.
        grounding: Where the evidence sits in the note.
        decision: Policy verdict.
        note: Source note.
        confidence: Score for the confidence extension.
        calibrated: Whether a fitted calibrator produced the score.
        resource_id: Deterministic id from the pipeline.
        value: Parsed value. When it carries a number, the resource gets a
            ``valueQuantity``; otherwise a ``valueString``.
        effective: Resolved date. When ``inferred``, the resource carries the
            ``effective-inferred`` extension so a reader can tell that
            fhirground supplied the time.
        include_quotes: Carry the evidence quote.
        mark_inferred_effective: Add the inferred flag. Configurable because
            some consumers reject unknown extensions.
        urls: Configured extension URLs.

    Returns:
        An ``Observation`` resource dict.

    Raises:
        MappingError: If the policy did not authorize emission.
    """
    if not decision.emit:
        raise MappingError(
            "build_observation called for a dropped mention "
            f"({decision.reason.value if decision.reason else 'unknown'})"
        )

    u = urls or ExtensionUrls()
    resource: dict[str, Any] = {
        "resourceType": "Observation",
        "id": resource_id,
        "status": DEFAULT_STATUS,
        "subject": {"reference": note.patient_ref},
        "code": resolution.as_codeable_concept(),
    }

    category = mention.obs_category
    if category is not None:
        resource["category"] = [
            {
                "coding": [
                    {
                        "system": CATEGORY_SYSTEM,
                        "code": category.value,
                        "display": _CATEGORY_DISPLAY[category],
                    }
                ]
            }
        ]

    if note.encounter_ref:
        resource["encounter"] = {"reference": note.encounter_ref}

    _apply_value(resource, mention, value)

    extensions: list[dict[str, Any]] = [
        confidence_extension(confidence, calibrated=calibrated, urls=u),
        evidence_extension(
            grounding,
            source_reference=note.source_reference,
            include_quote=include_quotes,
            urls=u,
        ),
    ]

    if effective is not None:
        resource["effectiveDateTime"] = effective.to_fhir()
        if effective.inferred and mark_inferred_effective:
            extensions.append(effective_inferred_extension(True, urls=u))

    resource["extension"] = extensions
    return resource


def _apply_value(resource: dict[str, Any], mention: Mention, value: ParsedValue | None) -> None:
    """Attach exactly one ``value[x]`` and any explanatory note.

    FHIR allows one value choice. Setting two would be invalid, so the order
    here is a strict preference: a parsed number wins, then the mention's own
    numeric field, then free text.
    """
    note_text = ""

    if value is not None and value.is_numeric:
        quantity: dict[str, Any] = {"value": value.number}
        if value.comparator is not None:
            quantity["comparator"] = value.comparator.value
        unit = resolve_unit(mention.value_unit)
        if unit is not None:
            quantity["unit"] = unit.text
            if unit.is_ucum:
                quantity["system"] = unit.system
                quantity["code"] = unit.code
        resource["valueQuantity"] = quantity
        note_text = value.note
    elif mention.value_number is not None:
        quantity = {"value": float(mention.value_number)}
        if mention.value_comparator is not None:
            quantity["comparator"] = mention.value_comparator.value
        unit = resolve_unit(mention.value_unit)
        if unit is not None:
            quantity["unit"] = unit.text
            if unit.is_ucum:
                quantity["system"] = unit.system
                quantity["code"] = unit.code
        resource["valueQuantity"] = quantity
    else:
        text = (value.text if value is not None else None) or mention.value_text
        if text:
            resource["valueString"] = text
        if value is not None:
            note_text = value.note

    if note_text:
        resource["note"] = [{"text": note_text}]


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
