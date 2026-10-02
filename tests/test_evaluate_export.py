"""The scorer and the output writers."""

from __future__ import annotations

import json

import pytest

from fhirground.backends import MockBackend
from fhirground.evaluate import (
    MatchSpec,
    read_resources,
    resources_match,
    score_resources,
)
from fhirground.export import FORMATS, clinval_label, write_outputs
from fhirground.pipeline import Pipeline, PipelineConfig
from tests.conftest import LOINC, SNOMED, make_note

VER_SYSTEM = "http://terminology.hl7.org/CodeSystem/condition-ver-status"


def condition(code="44054006", status="confirmed", text="type 2 diabetes"):
    return {
        "resourceType": "Condition",
        "id": f"c-{code}",
        "subject": {"reference": "Patient/1"},
        "code": {"coding": [{"system": SNOMED, "code": code}], "text": text},
        "verificationStatus": {"coding": [{"system": VER_SYSTEM, "code": status}]},
    }


def observation(code="4548-4", value=9.2):
    return {
        "resourceType": "Observation",
        "id": f"o-{code}",
        "status": "final",
        "subject": {"reference": "Patient/1"},
        "code": {"coding": [{"system": LOINC, "code": code}], "text": "a1c"},
        "valueQuantity": {"value": value, "unit": "%"},
    }


class TestMatching:
    def test_same_code_matches(self):
        assert resources_match(condition(), condition())

    def test_different_code_does_not(self):
        assert not resources_match(condition("44054006"), condition("38341003"))

    def test_different_type_does_not(self):
        assert not resources_match(condition(), observation())

    def test_status_must_agree(self):
        """A refuted Condition is not a confirmed one -- the point of §4.4."""
        assert not resources_match(condition(status="refuted"), condition(status="confirmed"))

    def test_status_can_be_ignored(self):
        spec = MatchSpec(require_status=False)
        assert resources_match(condition(status="refuted"), condition(status="confirmed"), spec)

    def test_numeric_tolerance(self):
        assert resources_match(
            observation(value=9.2),
            observation(value=9.21),
            MatchSpec(numeric_tolerance=0.01, exact_integers=False),
        )
        assert not resources_match(observation(value=9.2), observation(value=11.0))

    def test_integers_are_exact(self):
        """72 and 73 bpm are different observations."""
        assert not resources_match(observation(value=72), observation(value=73))

    def test_text_fallback_when_neither_is_coded(self):
        a = {"resourceType": "Condition", "code": {"text": "type 2 diabetes"}}
        b = {"resourceType": "Condition", "code": {"text": "type 2 diabetes"}}
        assert resources_match(a, b)

    def test_uncoded_prediction_against_coded_gold(self):
        uncoded = {"resourceType": "Condition", "code": {"text": "type 2 diabetes"}}
        assert not resources_match(uncoded, condition(), MatchSpec(require_code=True))
        assert resources_match(uncoded, condition(), MatchSpec(require_code=False))


class TestScoring:
    def test_perfect_score(self):
        resources = [condition(), observation()]
        report = score_resources(resources, resources)
        assert report.totals.tp == 2
        assert report.totals.precision == 1.0
        assert report.totals.recall == 1.0

    def test_false_positive_and_negative(self):
        report = score_resources([condition("44054006")], [condition("38341003")])
        assert report.per_type["Condition"].fp == 1
        assert report.per_type["Condition"].fn == 1

    def test_matching_is_one_to_one(self):
        """Emitting a finding twice costs a false positive."""
        report = score_resources([condition(), condition()], [condition()])
        assert report.per_type["Condition"].tp == 1
        assert report.per_type["Condition"].fp == 1

    def test_provenance_is_ignored(self):
        report = score_resources(
            [condition(), {"resourceType": "Provenance", "id": "p"}], [condition()]
        )
        assert report.totals.tp == 1
        assert report.totals.fp == 0

    def test_undefined_metrics_are_none_not_zero(self):
        report = score_resources([], [])
        assert report.totals.precision is None
        assert report.totals.recall is None
        assert report.totals.f1 is None

    def test_drop_reasons_are_carried(self):
        report = score_resources([condition()], [condition()], drop_reasons={"negated": 3})
        assert report.as_dict()["drop_reasons"]["negated"] == 3

    def test_report_includes_the_precision_caveat(self):
        """Precision rises when the pipeline emits less; say so."""
        assert "drop_reasons" in score_resources([], []).as_dict()["caveat"]

    def test_unmatched_lists_identify_resources(self):
        report = score_resources([condition("44054006")], [condition("38341003")])
        assert report.unmatched_predicted[0]["code"] == "44054006"
        assert report.unmatched_gold[0]["code"] == "38341003"

    def test_markdown_renders(self):
        md = score_resources([condition()], [condition()]).to_markdown()
        assert "Condition" in md
        assert "**all**" in md

    def test_json_safe(self):
        json.dumps(score_resources([condition()], [condition()]).as_dict())


class TestReadResources:
    def test_ndjson(self, tmp_path):
        path = tmp_path / "C.ndjson"
        path.write_text(json.dumps(condition()) + "\n" + json.dumps(condition("38341003")) + "\n")
        assert len(read_resources(path)) == 2

    def test_bundle(self, tmp_path):
        path = tmp_path / "b.json"
        path.write_text(
            json.dumps({"resourceType": "Bundle", "entry": [{"resource": condition()}]})
        )
        assert len(read_resources(path)) == 1

    def test_directory(self, tmp_path):
        (tmp_path / "a.ndjson").write_text(json.dumps(condition()) + "\n")
        (tmp_path / "b.json").write_text(json.dumps(observation()))
        assert len(read_resources(tmp_path)) == 2

    def test_missing_path(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_resources(tmp_path / "nope")


class TestClinvalLabel:
    def test_coded_label(self):
        assert clinval_label(condition()) == "Condition:44054006"

    def test_text_label_when_uncoded(self):
        resource = {"resourceType": "Condition", "code": {"text": "Fractured  FEMUR"}}
        assert clinval_label(resource) == "Condition:fractured femur"

    def test_medication_uses_its_own_field(self):
        resource = {
            "resourceType": "MedicationRequest",
            "medicationCodeableConcept": {
                "coding": [{"system": "x", "code": "860975"}],
                "text": "metformin",
            },
        }
        assert clinval_label(resource) == "MedicationRequest:860975"

    def test_unknown_concept(self):
        assert clinval_label({"resourceType": "Condition"}) == "Condition:unknown"


class TestWriteOutputs:
    @pytest.fixture
    def results(self, resolvers):
        notes = [
            make_note("Assessment: Type 2 diabetes. A1c 9.2% today.", note_id="n1"),
            make_note("Plan: start metformin 500 mg PO BID.", note_id="n2"),
        ]
        pipe = Pipeline(PipelineConfig(), backend=MockBackend(), resolvers=resolvers)
        return [pipe.run_note(n) for n in notes]

    def test_every_format(self, results, tmp_path):
        written = write_outputs(
            results, tmp_path, formats=FORMATS, model_id="m", prompt_version="v1"
        )
        assert (tmp_path / "bundles" / "n1.json").exists()
        assert (tmp_path / "ndjson" / "Condition.ndjson").exists()
        assert (tmp_path / "clinval_predictions.jsonl").exists()
        assert (tmp_path / "debug.jsonl").exists()
        assert (tmp_path / "run_summary.json").exists()
        assert set(written) >= {"bundles", "ndjson", "clinval", "debug", "summary"}

    def test_empty_ndjson_files_are_still_written(self, results, tmp_path):
        """So a consumer can tell 'none found' from 'not produced'."""
        write_outputs(results, tmp_path, formats=["ndjson"])
        for name in ("Condition", "MedicationRequest", "Observation", "Provenance"):
            assert (tmp_path / "ndjson" / f"{name}.ndjson").exists()

    def test_clinval_records_have_the_contract_shape(self, results, tmp_path):
        write_outputs(results, tmp_path, formats=["clinval"], model_id="m", prompt_version="v1")
        rows = [
            json.loads(line)
            for line in (tmp_path / "clinval_predictions.jsonl").read_text().splitlines()
            if line.strip()
        ]
        assert rows
        for row in rows:
            assert row["value"] is True
            assert ":" in row["label"]
            assert row["run"]["call_id"] == row["note_id"]
            assert 0.0 <= row["confidence"] <= 1.0
            assert "start" in row["span"]

    def test_clinval_output_is_readable_by_clinval_schema(self, results, tmp_path):
        """A shape check mirroring clinval's Prediction model."""
        write_outputs(results, tmp_path, formats=["clinval"])
        for line in (tmp_path / "clinval_predictions.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            assert {"system", "note_id", "label", "value"} <= set(row)
            assert isinstance(row["value"], bool)

    def test_debug_has_one_row_per_mention(self, results, tmp_path):
        write_outputs(results, tmp_path, formats=["debug"])
        rows = [
            json.loads(line)
            for line in (tmp_path / "debug.jsonl").read_text().splitlines()
            if line.strip()
        ]
        assert len(rows) == sum(len(r.debug) for r in results)
        for row in rows:
            assert "note_id" in row
            assert (row["resource_id"] is not None) != (row["drop_reason"] is not None)

    def test_summary_shape(self, results, tmp_path):
        write_outputs(
            results,
            tmp_path,
            formats=["summary"],
            cost_per_1k={"input_per_1k": 0.003, "output_per_1k": 0.015},
        )
        summary = json.loads((tmp_path / "run_summary.json").read_text())
        for key in ("n_notes", "resource_counts", "drop_counts", "versions", "cost_estimate"):
            assert key in summary
        assert summary["cost_estimate"]["total"] >= 0

    def test_cost_unknown_without_tokens(self, resolvers, tmp_path):
        from fhirground.backends.base import RawResponse

        class NoTokens(MockBackend):
            def extract(self, prompt, schema, *, temperature=0.0, n=1, **kwargs):
                out = super().extract(prompt, schema, temperature=temperature, n=n)
                return [RawResponse(text=r.text, model_id=r.model_id) for r in out]

        pipe = Pipeline(PipelineConfig(), backend=NoTokens(), resolvers=resolvers)
        results = [pipe.run_note(make_note("Assessment: Type 2 diabetes."))]
        write_outputs(
            results,
            tmp_path,
            formats=["summary"],
            cost_per_1k={"input_per_1k": 0.003, "output_per_1k": 0.015},
        )
        summary = json.loads((tmp_path / "run_summary.json").read_text())
        assert summary["cost_estimate"]["status"] == "unknown"

    def test_unknown_format_rejected(self, results, tmp_path):
        with pytest.raises(ValueError, match="unknown output format"):
            write_outputs(results, tmp_path, formats=["parquet"])

    def test_subset_of_formats(self, results, tmp_path):
        write_outputs(results, tmp_path, formats=["bundle"])
        assert (tmp_path / "bundles").exists()
        assert not (tmp_path / "debug.jsonl").exists()
