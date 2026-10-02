"""The assertion policy: every row of the SPEC §4.4 table."""

from __future__ import annotations

import pytest

from fhirground.assertion import (
    ACTIVE_MED_ACTIONS,
    DropReason,
    RequestStatus,
    VerificationStatus,
    decide,
)
from fhirground.schemas.mention import Assertion, MedAction, MentionType
from tests.conftest import make_mention


class TestConditionRows:
    """§4.4 condition rows."""

    def test_present_is_confirmed(self):
        d = decide(make_mention(assertion=Assertion.PRESENT))
        assert d.emit
        assert d.verification_status is VerificationStatus.CONFIRMED
        assert d.use_clinical_status_hint

    def test_possible_is_provisional(self):
        d = decide(make_mention(assertion=Assertion.POSSIBLE))
        assert d.emit
        assert d.verification_status is VerificationStatus.PROVISIONAL
        # A hedged finding has no established clinical course.
        assert not d.use_clinical_status_hint

    def test_negated_is_dropped_by_default(self):
        d = decide(make_mention(assertion=Assertion.NEGATED))
        assert not d.emit
        assert d.reason is DropReason.NEGATED

    def test_negated_becomes_refuted_when_asked(self):
        d = decide(make_mention(assertion=Assertion.NEGATED), emit_refuted=True)
        assert d.emit
        assert d.verification_status is VerificationStatus.REFUTED
        # con-5 forbids clinicalStatus beside entered-in-error, and a refuted
        # finding has no clinical course either.
        assert not d.use_clinical_status_hint

    def test_historical_is_confirmed_with_a_hint(self):
        d = decide(make_mention(assertion=Assertion.HISTORICAL))
        assert d.emit
        assert d.verification_status is VerificationStatus.CONFIRMED
        assert d.use_clinical_status_hint


class TestAnyTypeRows:
    """§4.4 rows that apply to every mention type."""

    @pytest.mark.parametrize("kind", list(MentionType))
    def test_family_is_dropped(self, kind):
        d = decide(make_mention(kind, assertion=Assertion.FAMILY))
        assert not d.emit
        assert d.reason is DropReason.FAMILY
        assert "FamilyMemberHistory" in d.note

    @pytest.mark.parametrize("kind", list(MentionType))
    def test_hypothetical_is_dropped(self, kind):
        d = decide(make_mention(kind, assertion=Assertion.HYPOTHETICAL))
        assert not d.emit
        assert d.reason is DropReason.HYPOTHETICAL


class TestMedicationRows:
    """§4.4 medication rows."""

    @pytest.mark.parametrize("action", sorted(ACTIVE_MED_ACTIONS, key=lambda a: a.value))
    def test_active_actions_are_active_orders(self, action):
        d = decide(make_mention(MentionType.MEDICATION, med_action=action))
        assert d.emit
        assert d.request_status is RequestStatus.ACTIVE

    def test_stop_is_a_stopped_order(self):
        d = decide(make_mention(MentionType.MEDICATION, med_action=MedAction.STOP))
        assert d.emit
        assert d.request_status is RequestStatus.STOPPED

    def test_mention_only_is_not_emitted_in_v1(self):
        d = decide(make_mention(MentionType.MEDICATION, med_action=MedAction.MENTION))
        assert not d.emit
        assert d.reason is DropReason.MED_MENTION_ONLY
        assert "MedicationStatement" in d.note

    def test_negated_medication_is_dropped(self):
        d = decide(
            make_mention(
                MentionType.MEDICATION,
                assertion=Assertion.NEGATED,
                med_action=MedAction.START,
            )
        )
        assert not d.emit
        assert d.reason is DropReason.NEGATED

    def test_no_action_is_not_guessed_into_an_order(self):
        """A MedicationRequest asserts somebody ordered the drug."""
        d = decide(make_mention(MentionType.MEDICATION, med_action=None))
        assert not d.emit
        assert d.reason is DropReason.MED_MENTION_ONLY
        assert "cannot be asserted" in d.note


class TestObservationRows:
    """§4.4 observation rows."""

    def test_present_is_emitted(self):
        d = decide(make_mention(MentionType.OBSERVATION, assertion=Assertion.PRESENT))
        assert d.emit

    def test_negated_is_dropped(self):
        d = decide(make_mention(MentionType.OBSERVATION, assertion=Assertion.NEGATED))
        assert not d.emit
        assert d.reason is DropReason.NEGATED

    @pytest.mark.parametrize("assertion", [Assertion.POSSIBLE, Assertion.HISTORICAL])
    def test_hedged_or_historical_is_not_a_final_result(self, assertion):
        """A measurement was either taken or not; `final` would overstate it."""
        d = decide(make_mention(MentionType.OBSERVATION, assertion=assertion))
        assert not d.emit
        assert d.reason is DropReason.UNSUPPORTED


class TestSectionOverride:
    """The section is better evidence than the sentence for family history."""

    def test_family_history_section_overrides_a_present_label(self):
        d = decide(make_mention(assertion=Assertion.PRESENT), section="family history")
        assert not d.emit
        assert d.reason is DropReason.FAMILY
        assert "family history" in d.note

    def test_override_can_be_disabled(self):
        d = decide(
            make_mention(assertion=Assertion.PRESENT),
            section="family history",
            respect_non_patient_sections=False,
        )
        assert d.emit

    def test_ordinary_sections_do_not_override(self):
        for section in ("assessment", "plan", "laboratory", None):
            d = decide(make_mention(assertion=Assertion.PRESENT), section=section)
            assert d.emit, section


class TestEveryRowCovered:
    def test_every_assertion_and_type_combination_has_a_verdict(self):
        """No combination falls through without a decision and a reason."""
        for kind in MentionType:
            for assertion in Assertion:
                for action in (None, *list(MedAction)):
                    if kind is not MentionType.MEDICATION and action is not None:
                        continue
                    mention = make_mention(kind, assertion=assertion, med_action=action)
                    d = decide(mention)
                    assert isinstance(d.emit, bool)
                    if d.emit:
                        assert d.note
                        if kind is MentionType.CONDITION:
                            assert d.verification_status is not None
                        if kind is MentionType.MEDICATION:
                            assert d.request_status is not None
                    else:
                        assert d.reason is not None
                        assert d.note
