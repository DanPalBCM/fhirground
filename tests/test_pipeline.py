"""The pipeline end to end, including the SPEC §5.2 definition-of-done example."""

from __future__ import annotations

import json

import pytest

from fhirground.assertion import DropReason
from fhirground.backends import MockBackend
from fhirground.exceptions import ConfigError
from fhirground.mapping.extensions import read_calibrated, read_confidence, read_evidence
from fhirground.pipeline import Pipeline, PipelineConfig, run_notes, summarize
from fhirground.schemas.mention import MentionType
from tests.conftest import make_note


def build(resolvers, **kwargs):
    """A pipeline with the mock backend."""
    config = PipelineConfig(**kwargs)
    return Pipeline(config, backend=MockBackend(), resolvers=resolvers)


class TestSpecExample:
    """SPEC §11: the §5.2 example must produce exactly these resources."""

    @pytest.fixture
    def result(self, spec_note, resolvers):
        return build(resolvers).run_note(spec_note)

    def test_exactly_three_clinical_resources(self, result):
        counts = result.resource_counts()
        assert counts.get("Condition") == 1
        assert counts.get("MedicationRequest") == 1
        assert counts.get("Observation") == 1
        assert counts.get("Provenance") == 1

    def test_the_condition_is_diabetes_and_coded(self, result):
        condition = result.by_type("Condition")[0]
        assert condition["code"]["coding"][0]["code"] == "44054006"
        assert condition["verificationStatus"]["coding"][0]["code"] == "confirmed"
        assert condition["subject"]["reference"] == "Patient/123"

    def test_the_observation_is_a1c_with_a_quantity(self, result):
        observation = result.by_type("Observation")[0]
        assert observation["code"]["coding"][0]["code"] == "4548-4"
        assert observation["valueQuantity"]["value"] == pytest.approx(9.2)
        assert observation["valueQuantity"]["unit"] == "%"
        assert observation["valueQuantity"]["system"] == "http://unitsofmeasure.org"
        assert observation["status"] == "final"

    def test_the_medication_is_metformin_active_order(self, result):
        medication = result.by_type("MedicationRequest")[0]
        assert medication["medicationCodeableConcept"]["coding"][0]["code"] == "860975"
        assert medication["status"] == "active"
        assert medication["intent"] == "order"
        assert "500" in medication["dosageInstruction"][0]["text"]

    def test_chest_pain_is_dropped_as_negated(self, result):
        drops = {m.text: reason for m, reason in result.dropped}
        assert "chest pain" in drops
        assert drops["chest pain"] is DropReason.NEGATED

    def test_hypertension_is_dropped_as_family_history(self, result):
        drops = {m.text: reason for m, reason in result.dropped}
        assert "hypertension" in drops
        assert drops["hypertension"] is DropReason.FAMILY

    def test_nothing_else_was_dropped(self, result):
        assert len(result.dropped) == 2

    def test_everything_validates(self, result):
        assert result.validation.ok, [str(f) for f in result.validation.errors]
        assert not result.validation.warnings

    def test_no_errors(self, result):
        assert result.ok
        assert result.errors == []

    def test_the_bundle_is_a_collection(self, result):
        assert result.bundle["resourceType"] == "Bundle"
        assert result.bundle["type"] == "collection"
        assert len(result.bundle["entry"]) == len(result.resources)


class TestGroundingAndEvidence:
    def test_every_resource_carries_checkable_evidence(self, spec_note, resolvers):
        result = build(resolvers).run_note(spec_note)
        for resource in result.resources:
            if resource["resourceType"] == "Provenance":
                continue
            evidence = read_evidence(resource)
            assert evidence
            for span in evidence:
                assert spec_note.text[span["start"] : span["end"]] == span["quote"]
                assert span["grounding"] in ("exact", "normalized", "fuzzy")
                assert span["source"] == "DocumentReference/docref-n1"

    def test_ungrounded_mentions_are_dropped(self, resolvers):
        """A fixture whose evidence is not in the note must not be emitted."""
        note = make_note("Assessment: Type 2 diabetes, poorly controlled.")
        fixtures = {
            MockBackend.fixture_key(note.text): [
                {
                    "mention_type": "condition",
                    "text": "pneumonia",
                    "evidence": "bilateral infiltrates on chest radiograph",
                    "assertion": "present",
                }
            ]
        }
        pipe = Pipeline(
            PipelineConfig(),
            backend=MockBackend(fixtures=fixtures, rules=False),
            resolvers=resolvers,
        )
        result = pipe.run_note(note)
        assert result.resource_counts().get("Condition") is None
        assert [r for _m, r in result.dropped] == [DropReason.UNGROUNDED]

    def test_quotes_can_be_omitted_while_offsets_remain(self, spec_note, resolvers):
        result = build(resolvers, include_quotes=False).run_note(spec_note)
        assert result.validation.ok
        for resource in result.resources:
            if resource["resourceType"] == "Provenance":
                continue
            for span in read_evidence(resource):
                assert "quote" not in span
                assert 0 <= span["start"] <= span["end"] <= len(spec_note.text)


class TestConfidence:
    def test_every_resource_has_a_confidence_in_range(self, spec_note, resolvers):
        result = build(resolvers).run_note(spec_note)
        for resource in result.resources:
            if resource["resourceType"] == "Provenance":
                continue
            score = read_confidence(resource)
            assert score is not None
            assert 0.0 <= score <= 1.0

    def test_uncalibrated_is_recorded_as_such(self, spec_note, resolvers):
        """The flag that separates a probability from a heuristic."""
        result = build(resolvers).run_note(spec_note)
        for resource in result.resources:
            if resource["resourceType"] == "Provenance":
                continue
            assert read_calibrated(resource) is False

    def test_min_confidence_filters(self, spec_note, resolvers):
        result = build(resolvers, min_confidence=0.999).run_note(spec_note)
        assert result.resource_counts().get("Condition") is None
        assert DropReason.BELOW_THRESHOLD in {r for _m, r in result.dropped}

    def test_terminology_component_absent_without_a_resolver(self, spec_note):
        from fhirground.terminology import ResolverConfig

        result = build(ResolverConfig()).run_note(spec_note)
        traces = [t for t in result.debug if t.components is not None]
        assert traces
        # With no resolver there is no `t` component -- absent, not zero.
        assert all(t.components.t is None for t in traces)

    def test_calibrated_flag_set_when_a_calibrator_is_supplied(self, spec_note, resolvers):
        from fhirground.confidence import Calibrator

        scores = [i / 40 for i in range(40)]
        labels = [0] * 20 + [1] * 20
        cal = Calibrator().fit(scores, labels, method="platt")
        pipe = Pipeline(
            PipelineConfig(), backend=MockBackend(), resolvers=resolvers, calibrator=cal
        )
        result = pipe.run_note(spec_note)
        for resource in result.resources:
            if resource["resourceType"] == "Provenance":
                continue
            assert read_calibrated(resource) is True


class TestProvenance:
    def test_one_provenance_per_note_targeting_every_resource(self, spec_note, resolvers):
        result = build(resolvers).run_note(spec_note)
        provenances = result.by_type("Provenance")
        assert len(provenances) == 1
        provenance = provenances[0]
        clinical = [
            f"{r['resourceType']}/{r['id']}"
            for r in result.resources
            if r["resourceType"] != "Provenance"
        ]
        targets = {t["reference"] for t in provenance["target"]}
        assert targets == set(clinical)

    def test_provenance_records_model_prompt_and_versions(self, spec_note, resolvers):
        result = build(resolvers).run_note(spec_note)
        label = result.by_type("Provenance")[0]["agent"][0]["who"]["display"]
        for expected in ("fhirground", "backend=mock", "model=", "prompt=", "fhir.resources"):
            assert expected in label

    def test_provenance_links_the_source_document(self, spec_note, resolvers):
        result = build(resolvers).run_note(spec_note)
        provenance = result.by_type("Provenance")[0]
        assert provenance["entity"][0]["what"]["reference"] == "DocumentReference/docref-n1"

    def test_no_provenance_when_nothing_was_emitted(self, resolvers):
        note = make_note("Patient seen for routine review. No findings.")
        pipe = Pipeline(PipelineConfig(), backend=MockBackend(rules=False), resolvers=resolvers)
        result = pipe.run_note(note)
        assert result.resources == []


class TestRepairLoop:
    def test_malformed_json_is_repaired(self, spec_note, resolvers):
        backend = MockBackend(fail_first_n=1)
        pipe = Pipeline(PipelineConfig(max_repairs=2), backend=backend, resolvers=resolvers)
        result = pipe.run_note(spec_note)
        assert result.ok
        assert result.stats["repairs"] == 1
        assert result.resource_counts().get("Condition") == 1

    def test_repairs_lower_the_validation_component(self, spec_note, resolvers):
        clean = build(resolvers).run_note(spec_note)
        repaired = Pipeline(
            PipelineConfig(max_repairs=2),
            backend=MockBackend(fail_first_n=1),
            resolvers=resolvers,
        ).run_note(spec_note)

        clean_v = next(t.components.v for t in clean.debug if t.components)
        repaired_v = next(t.components.v for t in repaired.debug if t.components)
        assert clean_v == 1.0
        assert repaired_v == 0.8

    def test_exhausting_repairs_reports_the_chunk_rather_than_crashing(self, spec_note, resolvers):
        pipe = Pipeline(
            PipelineConfig(max_repairs=1),
            backend=MockBackend(fail_first_n=99),
            resolvers=resolvers,
        )
        result = pipe.run_note(spec_note)
        assert not result.ok
        assert result.errors
        assert "failed after" in result.errors[0]
        assert result.resources == []


class TestDeduplication:
    def test_repeated_findings_merge_and_keep_every_span(self, resolvers):
        text = (
            "Assessment: Type 2 diabetes, poorly controlled.\n"
            "Plan: continue care for type 2 diabetes as discussed."
        )
        note = make_note(text)
        result = build(resolvers).run_note(note)
        conditions = result.by_type("Condition")
        assert len(conditions) == 1, "the same diagnosis twice must merge"
        evidence = read_evidence(conditions[0])
        assert len(evidence) >= 2, "a merged resource must keep every evidence span"
        for span in evidence:
            assert text[span["start"] : span["end"]] == span["quote"]

    def test_different_assertions_do_not_merge(self, resolvers):
        text = "Assessment: possible pneumonia. Later: pneumonia confirmed on imaging."
        note = make_note(text)
        result = build(resolvers).run_note(note)
        statuses = {
            c["verificationStatus"]["coding"][0]["code"] for c in result.by_type("Condition")
        }
        assert len(result.by_type("Condition")) >= 1
        assert statuses.issubset({"confirmed", "provisional"})


class TestConfigGuards:
    def test_overlap_must_be_smaller_than_chunk_size(self):
        with pytest.raises(ConfigError, match="must be smaller than max_chars"):
            PipelineConfig(max_chars=500, overlap_chars=500).validate_consistency()

    def test_samples_above_one_needs_temperature(self):
        """Identical samples would make self-consistency always 1.0."""
        with pytest.raises(ConfigError, match="identical"):
            PipelineConfig(samples=3, temperature=0.0).validate_consistency()

    def test_samples_with_temperature_is_allowed(self):
        PipelineConfig(samples=3, temperature=0.7).validate_consistency()


class TestMultipleNotes:
    def test_run_preserves_input_order(self, resolvers):
        notes = [
            make_note(f"Assessment: Type 2 diabetes. Note {i}.", note_id=f"n{i}") for i in range(6)
        ]
        results = run_notes(notes, MockBackend(), resolvers=resolvers, concurrency=3)
        assert [r.note.note_id for r in results] == [n.note_id for n in notes]

    def test_concurrency_one_runs_inline(self, resolvers):
        notes = [make_note("Assessment: Type 2 diabetes.", note_id="n0")]
        results = run_notes(notes, MockBackend(), resolvers=resolvers, concurrency=1)
        assert len(results) == 1

    def test_summary_shape(self, resolvers):
        notes = [
            make_note("Assessment: Type 2 diabetes. A1c 8.1% today.", note_id=f"n{i}")
            for i in range(3)
        ]
        results = run_notes(notes, MockBackend(), resolvers=resolvers, concurrency=1)
        summary = summarize(results)
        for key in (
            "n_notes",
            "n_mentions",
            "resource_counts",
            "drop_counts",
            "repairs",
            "tokens_in",
            "tokens_out",
            "versions",
        ):
            assert key in summary
        assert summary["n_notes"] == 3
        assert summary["versions"]["prompt"]
        json.dumps(summary)


class TestCache:
    def test_second_run_hits_the_cache(self, spec_note, resolvers, tmp_path):
        cache = tmp_path / "llm.sqlite"
        pipe = Pipeline(
            PipelineConfig(cache_path=cache), backend=MockBackend(), resolvers=resolvers
        )
        pipe.run_note(spec_note)
        assert pipe.cache is not None
        first = pipe.cache.stats()
        pipe.run_note(spec_note)
        second = pipe.cache.stats()
        assert second["hits"] > first["hits"]

    def test_cache_is_off_by_default(self, resolvers):
        assert build(resolvers).cache is None


class TestSectionHandling:
    def test_family_history_section_drops_a_bare_mention(self, resolvers):
        note = make_note(
            "Assessment:\nType 2 diabetes, stable.\n\nFamily History:\nHypertension.\n"
        )
        result = build(resolvers).run_note(note)
        codes = {
            c["code"]["coding"][0]["code"]
            for c in result.by_type("Condition")
            if c["code"].get("coding")
        }
        assert "44054006" in codes
        assert "38341003" not in codes, "a bare mention under Family History must drop"

    def test_override_can_be_disabled(self, resolvers):
        note = make_note(
            "Assessment:\nType 2 diabetes, stable.\n\nFamily History:\nHypertension.\n"
        )
        result = build(resolvers, respect_non_patient_sections=False).run_note(note)
        codes = {
            c["code"]["coding"][0]["code"]
            for c in result.by_type("Condition")
            if c["code"].get("coding")
        }
        assert "38341003" in codes


class TestTextOnlyConcepts:
    def test_unresolved_mentions_become_text_only(self):
        """Design principle 3: no resolver match means `code.text` only."""
        from fhirground.terminology import NullResolver, ResolverConfig

        note = make_note("Assessment: Type 2 diabetes, poorly controlled.")
        pipe = Pipeline(
            PipelineConfig(),
            backend=MockBackend(),
            resolvers=ResolverConfig(condition=NullResolver()),
        )
        result = pipe.run_note(note)
        condition = result.by_type("Condition")[0]
        assert "coding" not in condition["code"]
        assert condition["code"]["text"] == "type 2 diabetes"
        assert result.validation.ok


class TestObservationEffectiveTime:
    def test_resolved_date_is_used(self, spec_note, resolvers):
        result = build(resolvers).run_note(spec_note)
        observation = result.by_type("Observation")[0]
        assert observation["effectiveDateTime"] == "2026-03-14"

    def test_inferred_time_is_flagged(self, resolvers):
        from fhirground.mapping import ExtensionUrls

        note = make_note("Laboratory:\nA1c 9.2%.\n")
        result = build(resolvers).run_note(note)
        observation = result.by_type("Observation")[0]
        assert observation["effectiveDateTime"] == "2026-03-14"
        urls = ExtensionUrls()
        flags = [e for e in observation["extension"] if e["url"] == urls.effective_inferred]
        assert flags and flags[0]["valueBoolean"] is True

    def test_inference_can_be_disabled(self, resolvers):
        note = make_note("Laboratory:\nA1c 9.2%.\n")
        result = build(resolvers, infer_observation_effective=False).run_note(note)
        observation = result.by_type("Observation")[0]
        assert "effectiveDateTime" not in observation


class TestDebugTrace:
    def test_every_mention_has_exactly_one_outcome(self, spec_note, resolvers):
        result = build(resolvers).run_note(spec_note)
        assert result.debug
        for trace in result.debug:
            assert (trace.resource_id is not None) != (trace.drop_reason is not None)
            row = trace.as_dict()
            json.dumps(row)
            assert row["mention"]["text"]

    def test_trace_records_grounding_and_resolution(self, spec_note, resolvers):
        result = build(resolvers).run_note(spec_note)
        emitted = [t for t in result.debug if t.resource_id]
        assert emitted
        for trace in emitted:
            assert trace.grounding is not None
            assert trace.resolution is not None
            assert trace.components is not None
            assert trace.score is not None


class TestMentionTypeCoverage:
    def test_all_three_types_are_reachable(self, spec_note, resolvers):
        result = build(resolvers).run_note(spec_note)
        kinds = {t.mention.mention_type for t in result.debug}
        assert kinds == set(MentionType)
