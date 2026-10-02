"""Mapping to FHIR R4B: golden snapshots, invariants, and extension builders."""

from __future__ import annotations

import pytest
from fhir.resources.R4B.bundle import Bundle
from fhir.resources.R4B.condition import Condition
from fhir.resources.R4B.medicationrequest import MedicationRequest
from fhir.resources.R4B.observation import Observation
from fhir.resources.R4B.provenance import Provenance

from fhirground.assertion import decide
from fhirground.exceptions import MappingError
from fhirground.grounding import ground_mention
from fhirground.mapping import (
    ExtensionUrls,
    build_bundle,
    build_condition,
    build_medication_request,
    build_observation,
    build_provenance,
    confidence_extension,
    effective_inferred_extension,
    evidence_extension,
    read_calibrated,
    read_confidence,
    read_evidence,
)
from fhirground.normalize import parse_value, resolve_date
from fhirground.preprocess import chunk_note
from fhirground.schemas.mention import Assertion, ClinicalStatusHint, MedAction, MentionType
from tests.conftest import NOTE_DT, make_mention, make_note

NOTE_TEXT = (
    "Assessment: Type 2 diabetes, poorly controlled. A1c 9.2% today. "
    "Start metformin 500 mg PO BID. Denies chest pain."
)


@pytest.fixture
def note():
    return make_note(NOTE_TEXT, document_reference_id="docref-n1", encounter_ref="Encounter/e1")


def ground(evidence: str, text: str = NOTE_TEXT):
    return ground_mention(evidence, chunk_note(text, "n1", max_chars=4000).chunks, text)


class TestCondition:
    def test_golden_shape(self, note, resolvers):
        mention = make_mention()
        resource = build_condition(
            mention,
            resolvers.resolve(mention.text, MentionType.CONDITION),
            ground(mention.evidence),
            decide(mention),
            note,
            confidence=0.87,
            resource_id="c1",
        )
        Condition.model_validate(resource)
        assert resource["resourceType"] == "Condition"
        assert resource["id"] == "c1"
        assert resource["subject"] == {"reference": "Patient/123"}
        assert resource["encounter"] == {"reference": "Encounter/e1"}
        assert resource["code"]["coding"][0]["code"] == "44054006"
        assert resource["verificationStatus"]["coding"][0]["code"] == "confirmed"
        assert read_confidence(resource) == pytest.approx(0.87)
        assert len(read_evidence(resource)) == 1

    def test_clinical_status_omitted_without_a_hint(self, note, resolvers):
        """con-3 does not bind when category is absent, so omitting is correct."""
        mention = make_mention()
        resource = build_condition(
            mention,
            resolvers.resolve(mention.text, MentionType.CONDITION),
            ground(mention.evidence),
            decide(mention),
            note,
            confidence=0.5,
            resource_id="c1",
        )
        assert "clinicalStatus" not in resource

    def test_clinical_status_set_from_a_hint(self, note, resolvers):
        mention = make_mention(clinical_status_hint=ClinicalStatusHint.RESOLVED)
        resource = build_condition(
            mention,
            resolvers.resolve(mention.text, MentionType.CONDITION),
            ground(mention.evidence),
            decide(mention),
            note,
            confidence=0.5,
            resource_id="c1",
        )
        assert resource["clinicalStatus"]["coding"][0]["code"] == "resolved"
        Condition.model_validate(resource)

    def test_provisional_ignores_the_hint(self, note, resolvers):
        mention = make_mention(
            assertion=Assertion.POSSIBLE, clinical_status_hint=ClinicalStatusHint.ACTIVE
        )
        resource = build_condition(
            mention,
            resolvers.resolve(mention.text, MentionType.CONDITION),
            ground(mention.evidence),
            decide(mention),
            note,
            confidence=0.5,
            resource_id="c1",
        )
        assert resource["verificationStatus"]["coding"][0]["code"] == "provisional"
        assert "clinicalStatus" not in resource

    def test_refuted_when_asked(self, note, resolvers):
        mention = make_mention(assertion=Assertion.NEGATED)
        resource = build_condition(
            mention,
            resolvers.resolve(mention.text, MentionType.CONDITION),
            ground(mention.evidence),
            decide(mention, emit_refuted=True),
            note,
            confidence=0.5,
            resource_id="c1",
        )
        assert resource["verificationStatus"]["coding"][0]["code"] == "refuted"
        assert "clinicalStatus" not in resource, "con-5 territory: keep it off"

    def test_onset_string_carried(self, note, resolvers):
        mention = make_mention(onset_text="since 2019")
        resource = build_condition(
            mention,
            resolvers.resolve(mention.text, MentionType.CONDITION),
            ground(mention.evidence),
            decide(mention),
            note,
            confidence=0.5,
            resource_id="c1",
        )
        assert resource["onsetString"] == "since 2019"

    def test_category_without_a_hint_is_refused(self, note, resolvers):
        """include_category makes con-3 bind, so clinicalStatus is required."""
        mention = make_mention()
        with pytest.raises(MappingError, match="con-3"):
            build_condition(
                mention,
                resolvers.resolve(mention.text, MentionType.CONDITION),
                ground(mention.evidence),
                decide(mention),
                note,
                confidence=0.5,
                resource_id="c1",
                include_category=True,
            )

    def test_category_with_a_hint_is_allowed(self, note, resolvers):
        mention = make_mention(clinical_status_hint=ClinicalStatusHint.ACTIVE)
        resource = build_condition(
            mention,
            resolvers.resolve(mention.text, MentionType.CONDITION),
            ground(mention.evidence),
            decide(mention),
            note,
            confidence=0.5,
            resource_id="c1",
            include_category=True,
        )
        assert resource["category"][0]["coding"][0]["code"] == "problem-list-item"
        Condition.model_validate(resource)

    def test_dropped_mention_cannot_be_mapped(self, note, resolvers):
        mention = make_mention(assertion=Assertion.NEGATED)
        with pytest.raises(MappingError, match="dropped"):
            build_condition(
                mention,
                resolvers.resolve(mention.text, MentionType.CONDITION),
                ground(mention.evidence),
                decide(mention),
                note,
                confidence=0.5,
                resource_id="c1",
            )

    def test_text_only_concept(self, note):
        from fhirground.terminology import ResolverConfig

        mention = make_mention(text="fractured femur")
        resource = build_condition(
            mention,
            ResolverConfig().resolve(mention.text, MentionType.CONDITION),
            ground(mention.evidence),
            decide(mention),
            note,
            confidence=0.5,
            resource_id="c1",
        )
        assert resource["code"] == {"text": "fractured femur"}
        Condition.model_validate(resource)


class TestMedicationRequest:
    def test_golden_shape(self, note, resolvers, medication_mention):
        resource = build_medication_request(
            medication_mention,
            resolvers.resolve(medication_mention.text, MentionType.MEDICATION),
            ground(medication_mention.evidence),
            decide(medication_mention),
            note,
            confidence=0.9,
            resource_id="m1",
        )
        MedicationRequest.model_validate(resource)
        assert resource["status"] == "active"
        assert resource["intent"] == "order"
        assert resource["medicationCodeableConcept"]["coding"][0]["code"] == "860975"
        dosage = resource["dosageInstruction"][0]
        assert dosage["doseAndRate"][0]["doseQuantity"] == {
            "value": 500.0,
            "unit": "mg",
            "system": "http://unitsofmeasure.org",
            "code": "mg",
        }
        assert dosage["route"] == {"text": "PO"}
        assert "BID" in dosage["text"]

    def test_stopped_status(self, note, resolvers):
        mention = make_mention(
            MentionType.MEDICATION,
            text="metformin",
            evidence="Start metformin 500 mg PO BID",
            med_action=MedAction.STOP,
        )
        resource = build_medication_request(
            mention,
            resolvers.resolve(mention.text, MentionType.MEDICATION),
            ground(mention.evidence),
            decide(mention),
            note,
            confidence=0.5,
            resource_id="m1",
        )
        assert resource["status"] == "stopped"

    def test_authored_on_never_inferred(self, note, resolvers, medication_mention):
        """An order date must not be guessed from when the note was typed."""
        from fhirground.normalize.dates import inferred_from_note

        resource = build_medication_request(
            medication_mention,
            resolvers.resolve(medication_mention.text, MentionType.MEDICATION),
            ground(medication_mention.evidence),
            decide(medication_mention),
            note,
            confidence=0.5,
            resource_id="m1",
            authored_on=inferred_from_note(NOTE_DT),
        )
        assert "authoredOn" not in resource

    def test_authored_on_from_an_explicit_date(self, note, resolvers, medication_mention):
        resource = build_medication_request(
            medication_mention,
            resolvers.resolve(medication_mention.text, MentionType.MEDICATION),
            ground(medication_mention.evidence),
            decide(medication_mention),
            note,
            confidence=0.5,
            resource_id="m1",
            authored_on=resolve_date("2026-03-14", NOTE_DT),
        )
        assert resource["authoredOn"] == "2026-03-14"

    def test_unknown_dose_unit_has_no_ucum_code(self, note, resolvers):
        mention = make_mention(
            MentionType.MEDICATION,
            text="metformin",
            evidence="Start metformin 500 mg PO BID",
            med_action=MedAction.START,
            dose_value=2,
            dose_unit="scoops",
        )
        resource = build_medication_request(
            mention,
            resolvers.resolve(mention.text, MentionType.MEDICATION),
            ground(mention.evidence),
            decide(mention),
            note,
            confidence=0.5,
            resource_id="m1",
        )
        quantity = resource["dosageInstruction"][0]["doseAndRate"][0]["doseQuantity"]
        assert quantity["unit"] == "scoops"
        assert "system" not in quantity
        assert "code" not in quantity

    def test_dose_without_a_unit_is_not_structured(self, note, resolvers):
        mention = make_mention(
            MentionType.MEDICATION,
            text="metformin",
            evidence="Start metformin 500 mg PO BID",
            med_action=MedAction.START,
            dose_value=500,
            dose_unit=None,
        )
        resource = build_medication_request(
            mention,
            resolvers.resolve(mention.text, MentionType.MEDICATION),
            ground(mention.evidence),
            decide(mention),
            note,
            confidence=0.5,
            resource_id="m1",
        )
        assert "doseAndRate" not in resource["dosageInstruction"][0]


class TestObservation:
    def test_golden_shape(self, note, resolvers, observation_mention):
        resource = build_observation(
            observation_mention,
            resolvers.resolve(observation_mention.text, MentionType.OBSERVATION),
            ground(observation_mention.evidence),
            decide(observation_mention),
            note,
            confidence=0.8,
            resource_id="o1",
            value=parse_value(observation_mention.value_number),
            effective=resolve_date("today", NOTE_DT),
        )
        Observation.model_validate(resource)
        assert resource["status"] == "final"
        assert resource["code"]["coding"][0]["code"] == "4548-4"
        assert resource["category"][0]["coding"][0]["code"] == "laboratory"
        assert resource["valueQuantity"] == {
            "value": 9.2,
            "unit": "%",
            "system": "http://unitsofmeasure.org",
            "code": "%",
        }
        assert resource["effectiveDateTime"] == "2026-03-14"

    def test_comparator_carried(self, note, resolvers):
        mention = make_mention(
            MentionType.OBSERVATION,
            text="a1c",
            evidence="A1c 9.2% today",
            value_text="<0.01",
            value_unit="%",
        )
        resource = build_observation(
            mention,
            resolvers.resolve(mention.text, MentionType.OBSERVATION),
            ground(mention.evidence),
            decide(mention),
            note,
            confidence=0.5,
            resource_id="o1",
            value=parse_value("<0.01"),
        )
        assert resource["valueQuantity"]["comparator"] == "<"
        Observation.model_validate(resource)

    def test_range_adds_a_note(self, note, resolvers):
        mention = make_mention(
            MentionType.OBSERVATION, text="a1c", evidence="A1c 9.2% today", value_unit="%"
        )
        resource = build_observation(
            mention,
            resolvers.resolve(mention.text, MentionType.OBSERVATION),
            ground(mention.evidence),
            decide(mention),
            note,
            confidence=0.5,
            resource_id="o1",
            value=parse_value("5-10"),
        )
        assert resource["valueQuantity"]["value"] == 5.0
        assert "upper bound" in resource["note"][0]["text"]

    def test_pair_becomes_a_string_with_a_note(self, note, resolvers):
        mention = make_mention(
            MentionType.OBSERVATION, text="blood pressure", evidence="A1c 9.2% today"
        )
        resource = build_observation(
            mention,
            resolvers.resolve(mention.text, MentionType.OBSERVATION),
            ground(mention.evidence),
            decide(mention),
            note,
            confidence=0.5,
            resource_id="o1",
            value=parse_value("120/80"),
        )
        assert resource["valueString"] == "120/80"
        assert "valueQuantity" not in resource
        Observation.model_validate(resource)

    def test_only_one_value_choice_is_ever_set(self, note, resolvers, observation_mention):
        resource = build_observation(
            observation_mention,
            resolvers.resolve(observation_mention.text, MentionType.OBSERVATION),
            ground(observation_mention.evidence),
            decide(observation_mention),
            note,
            confidence=0.5,
            resource_id="o1",
            value=parse_value(observation_mention.value_number),
        )
        assert len([k for k in resource if k.startswith("value")]) == 1

    def test_inferred_effective_is_flagged(self, note, resolvers, observation_mention):
        from fhirground.normalize.dates import inferred_from_note

        urls = ExtensionUrls()
        resource = build_observation(
            observation_mention,
            resolvers.resolve(observation_mention.text, MentionType.OBSERVATION),
            ground(observation_mention.evidence),
            decide(observation_mention),
            note,
            confidence=0.5,
            resource_id="o1",
            effective=inferred_from_note(NOTE_DT),
        )
        flags = [e for e in resource["extension"] if e["url"] == urls.effective_inferred]
        assert flags and flags[0]["valueBoolean"] is True

    def test_unknown_unit_has_no_ucum_system(self, note, resolvers):
        mention = make_mention(
            MentionType.OBSERVATION,
            text="a1c",
            evidence="A1c 9.2% today",
            value_number=5.0,
            value_unit="widgets",
        )
        resource = build_observation(
            mention,
            resolvers.resolve(mention.text, MentionType.OBSERVATION),
            ground(mention.evidence),
            decide(mention),
            note,
            confidence=0.5,
            resource_id="o1",
            value=parse_value(5.0),
        )
        assert resource["valueQuantity"]["unit"] == "widgets"
        assert "system" not in resource["valueQuantity"]


class TestExtensions:
    def test_confidence_extension(self):
        ext = confidence_extension(0.87, calibrated=True)
        assert ext["valueDecimal"] == pytest.approx(0.87)
        assert ext["extension"][0]["valueBoolean"] is True

    @pytest.mark.parametrize("bad", [-0.1, 1.5])
    def test_out_of_range_confidence_rejected(self, bad):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            confidence_extension(bad)

    def test_evidence_extension_shape(self):
        g = ground("Denies chest pain")
        ext = evidence_extension(g, source_reference="DocumentReference/d1")
        urls = {sub["url"] for sub in ext["extension"]}
        assert urls == {"source", "start", "end", "quote", "grounding"}

    def test_quote_can_be_omitted_but_offsets_remain(self):
        g = ground("Denies chest pain")
        ext = evidence_extension(g, include_quote=False)
        urls = {sub["url"] for sub in ext["extension"]}
        assert "quote" not in urls
        assert {"start", "end"} <= urls

    def test_effective_inferred_extension(self):
        assert effective_inferred_extension(True)["valueBoolean"] is True

    def test_base_url_is_configurable_and_slash_normalized(self):
        a = ExtensionUrls("https://example.org/sd")
        b = ExtensionUrls("https://example.org/sd/")
        assert a.confidence == b.confidence == "https://example.org/sd/extraction-confidence"

    def test_readers_round_trip(self):
        g = ground("Denies chest pain")
        resource = {
            "extension": [
                confidence_extension(0.42, calibrated=False),
                evidence_extension(g),
            ]
        }
        assert read_confidence(resource) == pytest.approx(0.42)
        assert read_calibrated(resource) is False
        assert read_evidence(resource)[0]["start"] == g.start

    def test_readers_return_none_when_absent(self):
        assert read_confidence({}) is None
        assert read_calibrated({}) is None
        assert read_evidence({}) == []


class TestProvenanceAndBundle:
    def test_provenance_shape(self):
        resource = build_provenance(
            resource_id="p1",
            target_references=["Condition/c1", "Observation/o1"],
            source_reference="DocumentReference/d1",
            patient_ref="Patient/123",
            model_id="m",
            prompt_version="v1",
            package_version="1.0.0",
            fhir_library="fhir.resources 8.3.0 (R4B)",
            schema_digest="abc123",
            backend_name="mock",
        )
        Provenance.model_validate(resource)
        assert len(resource["target"]) == 2
        assert resource["entity"][0]["what"]["reference"] == "DocumentReference/d1"
        label = resource["agent"][0]["who"]["display"]
        assert "model=m" in label and "schema=abc123" in label

    def test_provenance_needs_targets(self):
        with pytest.raises(ValueError, match="at least one target"):
            build_provenance(
                resource_id="p1",
                target_references=[],
                source_reference=None,
                patient_ref="Patient/1",
                model_id="m",
                prompt_version="v",
                package_version="1",
                fhir_library="x",
            )

    def test_bundle_is_a_collection(self):
        bundle = build_bundle(
            [{"resourceType": "Condition", "id": "c1", "subject": {"reference": "Patient/1"}}],
            bundle_id="b1",
        )
        Bundle.model_validate(bundle)
        assert bundle["type"] == "collection"
        assert bundle["entry"][0]["resource"]["id"] == "c1"

    def test_empty_bundle_is_valid(self):
        Bundle.model_validate(build_bundle([], bundle_id="b1"))


class TestMedicationRequestEdgeCases:
    """Closing the §8.5 coverage requirement for the medication mapper."""

    def test_dropped_mention_is_refused(self, note, resolvers):
        mention = make_mention(
            MentionType.MEDICATION,
            text="metformin",
            evidence="Start metformin 500 mg PO BID",
            assertion=Assertion.NEGATED,
            med_action=MedAction.START,
        )
        with pytest.raises(MappingError, match="dropped mention"):
            build_medication_request(
                mention,
                resolvers.resolve(mention.text, MentionType.MEDICATION),
                ground(mention.evidence),
                decide(mention),
                note,
                confidence=0.5,
                resource_id="m1",
            )

    def test_missing_status_is_refused(self, note, resolvers, medication_mention):
        """A MedicationRequest with no status is not a valid order."""
        from fhirground.assertion import EmissionDecision

        broken = EmissionDecision(emit=True, request_status=None)
        with pytest.raises(MappingError, match="needs a status"):
            build_medication_request(
                medication_mention,
                resolvers.resolve(medication_mention.text, MentionType.MEDICATION),
                ground(medication_mention.evidence),
                broken,
                note,
                confidence=0.5,
                resource_id="m1",
            )

    def test_frequency_already_in_the_quote_is_not_duplicated(self, note, resolvers):
        mention = make_mention(
            MentionType.MEDICATION,
            text="metformin",
            evidence="Start metformin 500 mg PO BID",
            med_action=MedAction.START,
            frequency_text="BID",
        )
        resource = build_medication_request(
            mention,
            resolvers.resolve(mention.text, MentionType.MEDICATION),
            ground(mention.evidence),
            decide(mention),
            note,
            confidence=0.5,
            resource_id="m1",
        )
        assert resource["dosageInstruction"][0]["text"].count("BID") == 1

    def test_frequency_not_in_the_quote_is_appended(self, note, resolvers):
        mention = make_mention(
            MentionType.MEDICATION,
            text="metformin",
            evidence="Start metformin 500 mg PO BID",
            med_action=MedAction.START,
            frequency_text="with meals",
        )
        resource = build_medication_request(
            mention,
            resolvers.resolve(mention.text, MentionType.MEDICATION),
            ground(mention.evidence),
            decide(mention),
            note,
            confidence=0.5,
            resource_id="m1",
        )
        assert "(with meals)" in resource["dosageInstruction"][0]["text"]

    def test_add_evidence_appends_a_second_span(self, note, resolvers, medication_mention):
        from fhirground.mapping.medication_request import add_evidence

        resource = build_medication_request(
            medication_mention,
            resolvers.resolve(medication_mention.text, MentionType.MEDICATION),
            ground(medication_mention.evidence),
            decide(medication_mention),
            note,
            confidence=0.5,
            resource_id="m1",
        )
        before = len(read_evidence(resource))
        add_evidence(resource, ground("Type 2 diabetes"), note)
        assert len(read_evidence(resource)) == before + 1
        MedicationRequest.model_validate(resource)

    def test_add_evidence_on_the_other_mappers(self, note, resolvers, observation_mention):
        from fhirground.mapping.condition import add_evidence as add_cond
        from fhirground.mapping.observation import add_evidence as add_obs

        mention = make_mention()
        condition = build_condition(
            mention,
            resolvers.resolve(mention.text, MentionType.CONDITION),
            ground(mention.evidence),
            decide(mention),
            note,
            confidence=0.5,
            resource_id="c1",
        )
        add_cond(condition, ground("A1c 9.2% today"), note)
        assert len(read_evidence(condition)) == 2
        Condition.model_validate(condition)

        observation = build_observation(
            observation_mention,
            resolvers.resolve(observation_mention.text, MentionType.OBSERVATION),
            ground(observation_mention.evidence),
            decide(observation_mention),
            note,
            confidence=0.5,
            resource_id="o1",
        )
        add_obs(observation, ground("Type 2 diabetes"), note)
        assert len(read_evidence(observation)) == 2
        Observation.model_validate(observation)

    def test_observation_falls_back_to_the_mention_number(self, note, resolvers):
        """When no ParsedValue is supplied, the mention's own number is used."""
        mention = make_mention(
            MentionType.OBSERVATION,
            text="a1c",
            evidence="A1c 9.2% today",
            value_number=7.4,
            value_unit="%",
            value_comparator=None,
        )
        resource = build_observation(
            mention,
            resolvers.resolve(mention.text, MentionType.OBSERVATION),
            ground(mention.evidence),
            decide(mention),
            note,
            confidence=0.5,
            resource_id="o1",
            value=None,
        )
        assert resource["valueQuantity"]["value"] == pytest.approx(7.4)

    def test_observation_dropped_mention_is_refused(self, note, resolvers):
        mention = make_mention(
            MentionType.OBSERVATION,
            text="a1c",
            evidence="A1c 9.2% today",
            assertion=Assertion.NEGATED,
        )
        with pytest.raises(MappingError, match="dropped mention"):
            build_observation(
                mention,
                resolvers.resolve(mention.text, MentionType.OBSERVATION),
                ground(mention.evidence),
                decide(mention),
                note,
                confidence=0.5,
                resource_id="o1",
            )
