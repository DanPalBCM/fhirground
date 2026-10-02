"""Mention to FHIR R4B ``Condition``.

Note the signature of :func:`build_condition`: it takes a
:class:`~fhirground.terminology.base.Resolution`, **not** a code string. That
is the structural half of the no-invented-codes rule -- a caller cannot pass a
code that did not come from a resolver, because the parameter type will not
accept one.

FHIR invariants, quoted verbatim from the R4 StructureDefinition and verified
against it (see ``docs/decisions.md`` D-005):

* **con-3** (*warning*): "Condition.clinicalStatus SHALL be present if
  verificationStatus is not entered-in-error and category is
  problem-list-item". fhirground omits ``category`` by default, so the
  invariant does not bind and ``clinicalStatus`` may be absent. Note the
  polarity: the rule requires clinicalStatus when category **is**
  problem-list-item.
* **con-4** (*error*): "If condition is abated, then clinicalStatus must be
  either inactive, resolved, or remission". fhirground never sets
  ``abatement[x]`` in v1, so this is vacuously satisfied; the project rule
  checks it anyway.
* **con-5** (*error*): "Condition.clinicalStatus SHALL NOT be present if
  verification Status is entered-in-error" (the spacing typo is the spec's).
  fhirground never emits ``entered-in-error``.
"""

from __future__ import annotations

from typing import Any

from fhirground.assertion import EmissionDecision, VerificationStatus
from fhirground.exceptions import MappingError
from fhirground.grounding import Grounding
from fhirground.io import NoteInput
from fhirground.mapping.extensions import (
    ExtensionUrls,
    confidence_extension,
    evidence_extension,
)
from fhirground.schemas.mention import ClinicalStatusHint, Mention
from fhirground.terminology.base import Resolution

#: Terminology system for ``Condition.clinicalStatus``.
CLINICAL_STATUS_SYSTEM = "http://terminology.hl7.org/CodeSystem/condition-clinical"
#: Terminology system for ``Condition.verificationStatus``.
VERIFICATION_STATUS_SYSTEM = "http://terminology.hl7.org/CodeSystem/condition-ver-status"

#: Hint value -> (code, display) in the condition-clinical code system.
_CLINICAL_STATUS_CODES: dict[ClinicalStatusHint, tuple[str, str]] = {
    ClinicalStatusHint.ACTIVE: ("active", "Active"),
    ClinicalStatusHint.RESOLVED: ("resolved", "Resolved"),
    ClinicalStatusHint.INACTIVE: ("inactive", "Inactive"),
    ClinicalStatusHint.REMISSION: ("remission", "Remission"),
}

#: Verification status -> (code, display).
_VERIFICATION_CODES: dict[VerificationStatus, tuple[str, str]] = {
    VerificationStatus.CONFIRMED: ("confirmed", "Confirmed"),
    VerificationStatus.PROVISIONAL: ("provisional", "Provisional"),
    VerificationStatus.REFUTED: ("refuted", "Refuted"),
}

#: clinicalStatus codes that satisfy con-4 when an abatement is present.
CON4_ALLOWED_CLINICAL_STATUS = frozenset({"inactive", "resolved", "remission"})


def _status_concept(system: str, code: str, display: str) -> dict[str, Any]:
    return {"coding": [{"system": system, "code": code, "display": display}]}


def build_condition(
    mention: Mention,
    resolution: Resolution,
    grounding: Grounding,
    decision: EmissionDecision,
    note: NoteInput,
    *,
    confidence: float,
    calibrated: bool = False,
    resource_id: str,
    include_quotes: bool = True,
    include_category: bool = False,
    urls: ExtensionUrls | None = None,
) -> dict[str, Any]:
    """Build a ``Condition`` as a plain dict, ready for model validation.

    Args:
        mention: The source mention.
        resolution: The resolved concept. **A Resolution, never a code
            string** -- this is what makes an invented code impossible.
        grounding: Where the evidence sits in the note.
        decision: The assertion policy's verdict, which supplies
            ``verificationStatus``.
        note: The source note, for ``subject`` and ``encounter``.
        confidence: Score for the confidence extension.
        calibrated: Whether the score came from a fitted calibrator.
        resource_id: Deterministic id assigned by the pipeline.
        include_quotes: Carry the evidence quote in the resource.
        include_category: Set ``category`` to ``problem-list-item``. Off by
            default; turning it on makes con-3 bind, so the mapper then
            requires a ``clinicalStatus``.
        urls: Configured extension URLs.

    Returns:
        A ``Condition`` resource dict.

    Raises:
        MappingError: If the decision is not to emit, or if the combination
            would violate con-3 or con-5.
    """
    if not decision.emit:
        raise MappingError(
            f"build_condition called for a mention the policy dropped "
            f"({decision.reason.value if decision.reason else 'unknown'})"
        )
    if decision.verification_status is None:
        raise MappingError("a Condition needs a verificationStatus from the policy")

    u = urls or ExtensionUrls()
    ver_code, ver_display = _VERIFICATION_CODES[decision.verification_status]

    resource: dict[str, Any] = {
        "resourceType": "Condition",
        "id": resource_id,
        "subject": {"reference": note.patient_ref},
        "code": resolution.as_codeable_concept(),
        "verificationStatus": _status_concept(VERIFICATION_STATUS_SYSTEM, ver_code, ver_display),
    }

    if note.encounter_ref:
        resource["encounter"] = {"reference": note.encounter_ref}

    # clinicalStatus only from a hint, and only when the policy allows it.
    # con-5 forbids clinicalStatus beside entered-in-error; fhirground never
    # emits that status, and `refuted` carries no clinicalStatus either because
    # a refuted finding has no clinical course to describe.
    hint = mention.clinical_status_hint
    if decision.use_clinical_status_hint and hint is not None:
        code, display = _CLINICAL_STATUS_CODES[hint]
        resource["clinicalStatus"] = _status_concept(CLINICAL_STATUS_SYSTEM, code, display)

    if include_category:
        resource["category"] = [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/condition-category",
                        "code": "problem-list-item",
                        "display": "Problem List Item",
                    }
                ]
            }
        ]
        # con-3 binds once category is problem-list-item. Rather than emit a
        # resource that trips a spec warning, refuse and say why.
        if "clinicalStatus" not in resource and ver_code != "entered-in-error":
            raise MappingError(
                "include_category=True sets category to problem-list-item, which makes "
                "FHIR invariant con-3 require clinicalStatus, but this mention gave no "
                "clinical_status_hint. Either leave include_category off or supply a hint."
            )

    if mention.onset_text:
        resource["onsetString"] = mention.onset_text

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


def add_evidence(
    resource: dict[str, Any],
    grounding: Grounding,
    note: NoteInput,
    *,
    include_quotes: bool = True,
    urls: ExtensionUrls | None = None,
) -> None:
    """Append another evidence extension, for a deduplicated mention.

    When two mentions merge, both spans are kept. A merged resource that
    silently dropped one of its sources would be less auditable than two
    separate resources.
    """
    u = urls or ExtensionUrls()
    resource.setdefault("extension", []).append(
        evidence_extension(
            grounding,
            source_reference=note.source_reference,
            include_quote=include_quotes,
            urls=u,
        )
    )
