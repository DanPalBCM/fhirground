"""Assertion policy: which mentions become resources, and in what state.

This is SPEC §4.4 as executable code. It is the module that keeps "denies chest
pain" from becoming a Condition and "mother has hypertension" from becoming the
patient's diagnosis -- the failure mode that makes naive note-to-FHIR pipelines
dangerous rather than merely inaccurate.

The policy is deliberately a lookup, not a heuristic. Every row of the §4.4
table is a test case, and a mention whose combination is not in the table is
dropped with reason ``unsupported`` rather than guessed at.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from fhirground.schemas.mention import Assertion, MedAction, Mention, MentionType


class DropReason(StrEnum):
    """Why a mention did not become a resource.

    Counted and reported per note. A pipeline that drops most of what it finds
    is not working, so these counts are part of the run summary rather than
    buried in a debug file.
    """

    UNGROUNDED = "ungrounded"
    NEGATED = "negated"
    FAMILY = "family"
    HYPOTHETICAL = "hypothetical"
    MED_MENTION_ONLY = "med_mention_only"
    VALIDATION_FAILED = "validation_failed"
    BELOW_THRESHOLD = "below_threshold"
    UNSUPPORTED = "unsupported"
    NON_PATIENT_SECTION = "non_patient_section"


class VerificationStatus(StrEnum):
    """FHIR ``Condition.verificationStatus`` values used by v1."""

    CONFIRMED = "confirmed"
    PROVISIONAL = "provisional"
    REFUTED = "refuted"


class RequestStatus(StrEnum):
    """FHIR ``MedicationRequest.status`` values used by v1."""

    ACTIVE = "active"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class EmissionDecision:
    """What to do with one mention.

    Attributes:
        emit: Whether a resource should be produced.
        reason: Why not, when ``emit`` is False.
        verification_status: For Conditions.
        request_status: For MedicationRequests.
        use_clinical_status_hint: Whether the mapper may set
            ``clinicalStatus`` from the model's hint. False means omit the
            field rather than default it to ``active``.
        note: Short human-readable explanation, carried into the debug trace.
    """

    emit: bool
    reason: DropReason | None = None
    verification_status: VerificationStatus | None = None
    request_status: RequestStatus | None = None
    use_clinical_status_hint: bool = False
    note: str = ""


#: Medication actions that mean the prescriber is acting, so a
#: MedicationRequest is the right resource.
ACTIVE_MED_ACTIONS: frozenset[MedAction] = frozenset(
    {MedAction.START, MedAction.CONTINUE, MedAction.CHANGE}
)


def decide(
    mention: Mention,
    *,
    emit_refuted: bool = False,
    section: str | None = None,
    respect_non_patient_sections: bool = True,
) -> EmissionDecision:
    """Apply the §4.4 policy to one mention.

    Args:
        mention: The grounded mention.
        emit_refuted: When true, a negated condition is emitted with
            ``verificationStatus=refuted`` instead of being dropped. Off by
            default: most consumers want asserted findings, and a refuted
            Condition in a Bundle is easy to misread as a positive one.
        section: Name of the note section the mention came from.
        respect_non_patient_sections: When true, a mention from a
            family-history section is dropped as family even if the model
            labelled it ``present``. The section is better evidence than the
            sentence, because "hypertension" under *Family History* carries no
            negation cue for a model to notice.

    Returns:
        An :class:`EmissionDecision`.
    """
    from fhirground.preprocess.sections import is_non_patient_section

    if respect_non_patient_sections and is_non_patient_section(section):
        return EmissionDecision(
            emit=False,
            reason=DropReason.FAMILY,
            note=f"mention appeared under the {section!r} section",
        )

    # Assertions that are dropped regardless of mention type.
    if mention.assertion is Assertion.FAMILY:
        return EmissionDecision(
            emit=False,
            reason=DropReason.FAMILY,
            note="family history; FamilyMemberHistory is out of scope in v1",
        )
    if mention.assertion is Assertion.HYPOTHETICAL:
        return EmissionDecision(
            emit=False,
            reason=DropReason.HYPOTHETICAL,
            note="hypothetical or conditional mention",
        )

    if mention.mention_type is MentionType.CONDITION:
        return _decide_condition(mention, emit_refuted=emit_refuted)
    if mention.mention_type is MentionType.MEDICATION:
        return _decide_medication(mention)
    if mention.mention_type is MentionType.OBSERVATION:
        return _decide_observation(mention)

    return EmissionDecision(  # pragma: no cover - MentionType is exhaustive
        emit=False, reason=DropReason.UNSUPPORTED, note="unknown mention type"
    )


def _decide_condition(mention: Mention, *, emit_refuted: bool) -> EmissionDecision:
    """§4.4 condition rows."""
    if mention.assertion is Assertion.NEGATED:
        if emit_refuted:
            return EmissionDecision(
                emit=True,
                verification_status=VerificationStatus.REFUTED,
                use_clinical_status_hint=False,
                note="negated condition emitted as refuted",
            )
        return EmissionDecision(emit=False, reason=DropReason.NEGATED, note="negated condition")

    if mention.assertion is Assertion.POSSIBLE:
        return EmissionDecision(
            emit=True,
            verification_status=VerificationStatus.PROVISIONAL,
            use_clinical_status_hint=False,
            note="hedged condition emitted as provisional",
        )

    if mention.assertion is Assertion.HISTORICAL:
        # clinicalStatus comes from the hint if the model gave one (often
        # `resolved`), and is otherwise omitted. FHIR invariant con-3 forbids
        # clinicalStatus on a refuted condition, and guessing `active` for a
        # historical finding would be worse than saying nothing.
        return EmissionDecision(
            emit=True,
            verification_status=VerificationStatus.CONFIRMED,
            use_clinical_status_hint=True,
            note="historical condition",
        )

    if mention.assertion is Assertion.PRESENT:
        return EmissionDecision(
            emit=True,
            verification_status=VerificationStatus.CONFIRMED,
            use_clinical_status_hint=True,
            note="present condition",
        )

    return EmissionDecision(  # pragma: no cover
        emit=False, reason=DropReason.UNSUPPORTED, note="unhandled condition assertion"
    )


def _decide_medication(mention: Mention) -> EmissionDecision:
    """§4.4 medication rows."""
    if mention.assertion is Assertion.NEGATED:
        return EmissionDecision(emit=False, reason=DropReason.NEGATED, note="negated medication")

    action = mention.med_action
    if action is MedAction.STOP:
        return EmissionDecision(
            emit=True,
            request_status=RequestStatus.STOPPED,
            note="medication stopped",
        )
    if action in ACTIVE_MED_ACTIONS:
        return EmissionDecision(
            emit=True,
            request_status=RequestStatus.ACTIVE,
            note=f"medication {action.value}",
        )
    if action is MedAction.MENTION:
        return EmissionDecision(
            emit=False,
            reason=DropReason.MED_MENTION_ONLY,
            note="patient-reported medication; MedicationStatement is v1.1",
        )

    # No action given. v1 does not guess: a MedicationRequest asserts that
    # somebody ordered the drug, and a bare mention does not support that.
    return EmissionDecision(
        emit=False,
        reason=DropReason.MED_MENTION_ONLY,
        note="no med_action given, so an order cannot be asserted",
    )


def _decide_observation(mention: Mention) -> EmissionDecision:
    """§4.4 observation rows."""
    if mention.assertion is Assertion.PRESENT:
        return EmissionDecision(emit=True, note="present observation")
    if mention.assertion is Assertion.NEGATED:
        return EmissionDecision(emit=False, reason=DropReason.NEGATED, note="negated observation")
    # historical / possible observations: a measurement is either taken or not,
    # so a hedged observation is not emitted as a final result.
    return EmissionDecision(
        emit=False,
        reason=DropReason.UNSUPPORTED,
        note=f"observation with assertion {mention.assertion.value} is not emitted in v1",
    )
