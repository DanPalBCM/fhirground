"""The synthetic benchmark: determinism, gold correctness, and PHI safety."""

from __future__ import annotations

import json
import re

import pytest

from fhirground.exceptions import InputError
from fhirground.synth import (
    CONDITION_BANK,
    MEDICATION_BANK,
    OBSERVATION_BANK,
    build_benchmark,
    generate_records,
    records_from_synthea,
    value_sets_from_records,
    write_benchmark,
)


class TestDeterminism:
    def test_same_seed_same_records(self):
        a = generate_records(8, seed=3)
        b = generate_records(8, seed=3)
        assert [r.patient_id for r in a] == [r.patient_id for r in b]
        assert [r.note_datetime for r in a] == [r.note_datetime for r in b]

    def test_different_seeds_differ(self):
        a = generate_records(8, seed=1)
        b = generate_records(8, seed=2)
        assert [r.note_datetime for r in a] != [r.note_datetime for r in b]

    def test_same_seed_same_notes(self):
        a, _ = build_benchmark(n_notes=6, seed=0)
        b, _ = build_benchmark(n_notes=6, seed=0)
        assert [n.text for n in a] == [n.text for n in b]

    def test_written_files_are_byte_identical(self, tmp_path):
        write_benchmark(tmp_path / "a", n_notes=5, seed=0)
        write_benchmark(tmp_path / "b", n_notes=5, seed=0)
        for name in ("notes.jsonl", "gold/Condition.ndjson", "valuesets/conditions.csv"):
            assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()

    def test_zero_patients_rejected(self):
        with pytest.raises(ValueError, match="at least 1"):
            generate_records(0)


class TestGoldCorrectness:
    def test_gold_is_generated_from_the_same_records(self):
        """Exact by construction, not annotated."""
        _notes, rendered = build_benchmark(n_notes=10, seed=0)
        for note in rendered:
            assert note.gold, "every note must have gold"
            for resource in note.gold:
                assert resource["subject"]["reference"] == note.patient_ref
                assert resource["resourceType"] in ("Condition", "MedicationRequest", "Observation")

    def test_gold_resources_parse_as_fhir(self):
        from fhirground.validate import parse_with_models

        _notes, rendered = build_benchmark(n_notes=8, seed=0)
        for note in rendered:
            for resource in note.gold:
                assert parse_with_models(resource).ok, resource

    def test_gold_codings_have_a_system_and_code(self):
        _notes, rendered = build_benchmark(n_notes=8, seed=0)
        for note in rendered:
            for resource in note.gold:
                concept = resource.get("code") or resource["medicationCodeableConcept"]
                coding = concept["coding"][0]
                assert coding["system"].startswith("http")
                assert coding["code"]


class TestDistractors:
    def test_negated_and_family_distractors_are_injected(self):
        _notes, rendered = build_benchmark(n_notes=20, seed=0)
        kinds = {d["kind"] for note in rendered for d in note.distractors}
        assert "negated" in kinds
        assert "family" in kinds

    def test_distractors_are_recorded_with_what_to_expect(self):
        _notes, rendered = build_benchmark(n_notes=10, seed=0)
        for note in rendered:
            for d in note.distractors:
                assert d["expect"] in ("dropped", "provisional")
                assert d["text"]

    def test_distractors_can_be_disabled(self):
        _notes, rendered = build_benchmark(n_notes=6, seed=0, inject_distractors=False)
        assert all(not note.distractors for note in rendered)

    def test_negated_distractors_are_not_in_the_gold(self):
        """Otherwise the benchmark would reward emitting them."""
        _notes, rendered = build_benchmark(n_notes=15, seed=0)
        for note in rendered:
            gold_texts = {
                (r.get("code") or r["medicationCodeableConcept"])["text"].lower() for r in note.gold
            }
            for d in note.distractors:
                if d["kind"] == "negated":
                    assert d["text"].lower() not in gold_texts

    def test_the_pipeline_drops_them(self, resolvers):
        """The end-to-end point of the distractors."""
        from fhirground.assertion import DropReason
        from fhirground.backends import MockBackend
        from fhirground.pipeline import Pipeline, PipelineConfig

        notes, _rendered = build_benchmark(n_notes=10, seed=0)
        pipe = Pipeline(PipelineConfig(), backend=MockBackend(), resolvers=resolvers)
        reasons = set()
        for note in notes:
            result = pipe.run_note(note)
            reasons |= {r for _m, r in result.dropped}
        assert DropReason.NEGATED in reasons
        assert DropReason.FAMILY in reasons


class TestNoRealData:
    """PHI safety is structural: there is no source to leak from."""

    def test_identifiers_are_sequential_placeholders(self):
        records = generate_records(10, seed=0)
        assert all(re.fullmatch(r"synth-\d{5}", r.patient_id) for r in records)

    def test_note_ids_are_placeholders(self):
        notes, _ = build_benchmark(n_notes=6, seed=0)
        assert all(re.fullmatch(r"synth-note-\d{5}", n.note_id) for n in notes)

    def test_note_text_comes_from_the_banks(self):
        """Every clinical term in a note traces to a declared bank entry."""
        _notes, rendered = build_benchmark(n_notes=15, seed=0)
        vocabulary = set()
        for bank in (CONDITION_BANK, MEDICATION_BANK, OBSERVATION_BANK):
            for entry in bank:
                vocabulary.add(entry["display"].lower())
                for abbrev in entry.get("abbrev") or ():
                    vocabulary.add(abbrev.lower())
        assert vocabulary
        for note in rendered:
            assert note.text.strip()
            # The note mentions at least one declared concept and nothing
            # resembling a real identifier.
            assert not re.search(r"\b\d{3}-\d{2}-\d{4}\b", note.text)
            assert not re.search(r"\bMRN\b", note.text)

    def test_no_long_digit_runs_that_could_be_identifiers(self):
        _notes, rendered = build_benchmark(n_notes=15, seed=0)
        for note in rendered:
            # Years are four digits and legitimate; nothing longer appears.
            assert not re.search(r"\d{5,}", note.text)

    def test_dates_derive_from_the_fixed_epoch(self):
        from fhirground.synth import EPOCH

        records = generate_records(10, seed=0)
        assert all(r.note_datetime >= EPOCH.replace(tzinfo=None) for r in records)


class TestValueSets:
    def test_cover_exactly_the_codes_used(self):
        records = generate_records(10, seed=0)
        value_sets = value_sets_from_records(records)
        used = {c["code"] for r in records for c in r.conditions}
        assert {row["code"] for row in value_sets["conditions"]} == used

    def test_every_row_has_the_required_columns(self):
        value_sets = value_sets_from_records(generate_records(5, seed=0))
        for rows in value_sets.values():
            for row in rows:
                assert row["system"] and row["code"] and row["display"]

    def test_written_csvs_load_back(self, tmp_path):
        from fhirground.terminology import LocalValueSet

        written = write_benchmark(tmp_path, n_notes=6, seed=0)
        vs = LocalValueSet.from_csv(written["valueset_conditions"])
        assert len(vs) > 0


class TestWriteBenchmark:
    def test_artifacts(self, tmp_path):
        written = write_benchmark(tmp_path, n_notes=8, seed=0)
        for key in ("notes", "manifest", "gold_Condition", "valueset_conditions"):
            assert written[key].exists()

    def test_manifest_records_provenance(self, tmp_path):
        write_benchmark(tmp_path, n_notes=8, seed=0)
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["source"] == "generated"
        assert manifest["n_notes"] == 8
        assert manifest["seed"] == 0
        assert manifest["synthea_version_documented"]
        assert "No real patient record" in manifest["note"]
        assert manifest["distractors"]

    def test_notes_are_readable(self, tmp_path):
        from fhirground.io import read_notes_jsonl

        written = write_benchmark(tmp_path, n_notes=5, seed=0)
        notes = read_notes_jsonl(written["notes"])
        assert len(notes) == 5
        assert all(n.patient_ref.startswith("Patient/") for n in notes)

    def test_gold_is_readable_by_the_scorer(self, tmp_path):
        from fhirground.evaluate import read_resources

        write_benchmark(tmp_path, n_notes=5, seed=0)
        assert read_resources(tmp_path / "gold")


class TestSyntheaReading:
    def test_reads_a_synthea_shaped_bundle(self, tmp_path):
        bundle = {
            "resourceType": "Bundle",
            "entry": [
                {"resource": {"resourceType": "Patient", "id": "p1"}},
                {
                    "resource": {
                        "resourceType": "Condition",
                        "id": "c1",
                        "subject": {"reference": "Patient/p1"},
                        "code": {
                            "coding": [
                                {
                                    "system": "http://snomed.info/sct",
                                    "code": "44054006",
                                    "display": "Type 2 diabetes mellitus",
                                }
                            ]
                        },
                    }
                },
                {
                    "resource": {
                        "resourceType": "Observation",
                        "id": "o1",
                        "subject": {"reference": "Patient/p1"},
                        "code": {
                            "coding": [
                                {
                                    "system": "http://loinc.org",
                                    "code": "4548-4",
                                    "display": "Hemoglobin A1c",
                                }
                            ]
                        },
                        "valueQuantity": {"value": 8.1, "unit": "%"},
                    }
                },
            ],
        }
        path = tmp_path / "p1.json"
        path.write_text(json.dumps(bundle))
        records = records_from_synthea(path)
        assert len(records) == 1
        assert records[0].conditions[0]["code"] == "44054006"
        assert records[0].observations[0]["value"] == 8.1

    def test_renders_notes_from_synthea_records(self, tmp_path):
        bundle = {
            "resourceType": "Bundle",
            "entry": [
                {"resource": {"resourceType": "Patient", "id": "p1"}},
                {
                    "resource": {
                        "resourceType": "Condition",
                        "id": "c1",
                        "subject": {"reference": "Patient/p1"},
                        "code": {
                            "coding": [
                                {
                                    "system": "http://snomed.info/sct",
                                    "code": "44054006",
                                    "display": "Type 2 diabetes mellitus",
                                }
                            ]
                        },
                    }
                },
            ],
        }
        (tmp_path / "p1.json").write_text(json.dumps(bundle))
        notes, rendered = build_benchmark(n_notes=1, seed=0, synthea_dir=tmp_path)
        assert len(notes) == 1
        assert rendered[0].gold

    def test_nothing_usable_explains_what_to_do(self, tmp_path):
        (tmp_path / "x.json").write_text(json.dumps({"resourceType": "Patient", "id": "p"}))
        with pytest.raises(InputError, match="--synthea-dir"):
            records_from_synthea(tmp_path)

    def test_missing_path(self, tmp_path):
        with pytest.raises(InputError, match="not found"):
            records_from_synthea(tmp_path / "nope")
