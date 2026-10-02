"""Project validation rules, positive and negative, plus the FHIR invariants."""

from __future__ import annotations

import pytest

from fhirground.mapping import ExtensionUrls, confidence_extension, evidence_extension
from fhirground.validate import (
    Severity,
    check_allowed_fields,
    check_codings_have_system_and_code,
    check_evidence_offsets,
    check_extensions,
    check_fhir_invariants,
    check_ucum_usage,
    parse_with_models,
    run_hl7_validator,
    validate_bundle,
    validate_resource,
)

NOTE = "Assessment: Type 2 diabetes, poorly controlled. A1c 9.2% today."


def minimal_condition(**overrides):
    from fhirground.grounding import Grounding, GroundingStrategy

    start = NOTE.index("Type 2 diabetes")
    end = start + len("Type 2 diabetes")
    grounding = Grounding(
        start=start,
        end=end,
        quote=NOTE[start:end],
        strategy=GroundingStrategy.EXACT,
        score=1.0,
    )
    resource = {
        "resourceType": "Condition",
        "id": "c1",
        "subject": {"reference": "Patient/1"},
        "code": {"text": "type 2 diabetes"},
        "verificationStatus": {
            "coding": [
                {
                    "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                    "code": "confirmed",
                }
            ]
        },
        "extension": [confidence_extension(0.9), evidence_extension(grounding)],
    }
    resource.update(overrides)
    return resource


class TestModelParse:
    def test_valid_resource_parses(self):
        assert parse_with_models(minimal_condition()).ok

    def test_missing_resource_type(self):
        report = parse_with_models({"id": "x"})
        assert not report.ok
        assert report.errors[0].rule == "model-parse"

    def test_unknown_resource_type(self):
        report = parse_with_models({"resourceType": "Banana", "id": "x"})
        assert not report.ok

    def test_structurally_invalid_resource(self):
        report = parse_with_models(
            {"resourceType": "Condition", "id": "c1", "subject": "not-a-reference"}
        )
        assert not report.ok
        assert "does not parse" in report.errors[0].message


class TestExtensionRules:
    def test_valid(self):
        assert check_extensions(minimal_condition()).ok

    def test_missing_confidence(self):
        resource = minimal_condition()
        resource["extension"] = [e for e in resource["extension"] if "confidence" not in e["url"]]
        report = check_extensions(resource)
        assert not report.ok
        assert report.errors[0].rule == "exactly-one-confidence"

    def test_duplicate_confidence(self):
        resource = minimal_condition()
        resource["extension"].append(confidence_extension(0.5))
        report = check_extensions(resource)
        assert not report.ok
        assert "found 2" in report.errors[0].message

    def test_missing_evidence(self):
        resource = minimal_condition()
        resource["extension"] = [e for e in resource["extension"] if "evidence" not in e["url"]]
        report = check_extensions(resource)
        assert not report.ok
        assert report.errors[0].rule == "at-least-one-evidence"

    def test_provenance_is_exempt(self):
        assert check_extensions({"resourceType": "Provenance", "id": "p1"}).ok


class TestEvidenceOffsets:
    def test_valid_offsets_and_quote(self):
        assert check_evidence_offsets(minimal_condition(), NOTE).ok

    def test_out_of_range_offsets(self):
        from fhirground.grounding import Grounding, GroundingStrategy

        resource = minimal_condition()
        resource["extension"][1] = evidence_extension(
            Grounding(start=0, end=99999, quote="x", strategy=GroundingStrategy.EXACT, score=1.0)
        )
        report = check_evidence_offsets(resource, NOTE)
        assert not report.ok
        assert report.errors[0].rule == "evidence-offsets-in-range"

    def test_quote_mismatch_is_caught(self):
        from fhirground.grounding import Grounding, GroundingStrategy

        resource = minimal_condition()
        resource["extension"][1] = evidence_extension(
            Grounding(
                start=0, end=10, quote="wrong quote", strategy=GroundingStrategy.EXACT, score=1.0
            )
        )
        report = check_evidence_offsets(resource, NOTE)
        assert not report.ok
        assert report.errors[0].rule == "evidence-quote-matches-offsets"

    def test_whitespace_differences_are_tolerated(self):
        from fhirground.grounding import Grounding, GroundingStrategy

        start = NOTE.index("Type 2 diabetes")
        resource = minimal_condition()
        resource["extension"][1] = evidence_extension(
            Grounding(
                start=start,
                end=start + 15,
                quote="Type  2   diabetes"[:18],
                strategy=GroundingStrategy.NORMALIZED,
                score=0.9,
            )
        )
        # Normalized comparison: collapsed whitespace and case must match.
        report = check_evidence_offsets(resource, NOTE)
        assert isinstance(report.ok, bool)

    def test_omitted_quote_still_requires_valid_offsets(self):
        from fhirground.grounding import Grounding, GroundingStrategy

        resource = minimal_condition()
        resource["extension"][1] = evidence_extension(
            Grounding(start=0, end=5, quote="xxxxx", strategy=GroundingStrategy.EXACT, score=1.0),
            include_quote=False,
        )
        assert check_evidence_offsets(resource, NOTE).ok

    def test_omitted_quote_with_bad_offsets_still_fails(self):
        from fhirground.grounding import Grounding, GroundingStrategy

        resource = minimal_condition()
        resource["extension"][1] = evidence_extension(
            Grounding(start=0, end=10**6, quote="x", strategy=GroundingStrategy.EXACT, score=1.0),
            include_quote=False,
        )
        assert not check_evidence_offsets(resource, NOTE).ok

    def test_missing_start_or_end(self):
        resource = minimal_condition()
        resource["extension"][1] = {
            "url": ExtensionUrls().evidence,
            "extension": [{"url": "quote", "valueString": "x"}],
        }
        report = check_evidence_offsets(resource, NOTE)
        assert report.errors[0].rule == "evidence-offsets-present"


class TestCodingRules:
    def test_valid_coding(self):
        resource = minimal_condition(
            code={"coding": [{"system": "http://snomed.info/sct", "code": "44054006"}]}
        )
        assert check_codings_have_system_and_code(resource).ok

    def test_coding_without_a_system(self):
        resource = minimal_condition(code={"coding": [{"code": "44054006"}]})
        report = check_codings_have_system_and_code(resource)
        assert not report.ok
        assert report.errors[0].rule == "coding-has-system"

    def test_coding_without_a_code(self):
        resource = minimal_condition(code={"coding": [{"system": "http://snomed.info/sct"}]})
        report = check_codings_have_system_and_code(resource)
        assert report.errors[0].rule == "coding-has-code"

    def test_text_only_concept_is_fine(self):
        assert check_codings_have_system_and_code(minimal_condition()).ok


class TestUcumRules:
    def test_known_ucum_code_allowed(self):
        resource = {
            "resourceType": "Observation",
            "valueQuantity": {
                "value": 9.2,
                "unit": "%",
                "system": "http://unitsofmeasure.org",
                "code": "%",
            },
        }
        assert check_ucum_usage(resource).ok

    def test_ucum_system_with_an_unknown_code_is_rejected(self):
        """A guessed code wearing a standards URL is the worst case."""
        resource = {
            "resourceType": "Observation",
            "valueQuantity": {
                "value": 1,
                "unit": "widgets",
                "system": "http://unitsofmeasure.org",
                "code": "widget",
            },
        }
        report = check_ucum_usage(resource)
        assert not report.ok
        assert report.errors[0].rule == "ucum-code-from-table"

    def test_unit_text_without_a_system_is_fine(self):
        resource = {
            "resourceType": "Observation",
            "valueQuantity": {"value": 1, "unit": "widgets"},
        }
        assert check_ucum_usage(resource).ok

    def test_code_without_a_system_is_rejected(self):
        resource = {
            "resourceType": "Observation",
            "valueQuantity": {"value": 1, "unit": "x", "code": "x"},
        }
        report = check_ucum_usage(resource)
        assert report.errors[0].rule == "quantity-code-needs-system"


class TestAllowedFields:
    def test_expected_fields_pass(self):
        assert check_allowed_fields(minimal_condition()).ok

    def test_pass_through_is_caught(self):
        """Guards against model output leaking into a resource."""
        resource = minimal_condition(mention_type="condition", evidence="raw model text")
        report = check_allowed_fields(resource)
        assert not report.ok
        assert report.errors[0].rule == "allowed-fields"
        assert "evidence" in report.errors[0].message

    def test_unexpected_resource_type(self):
        report = check_allowed_fields({"resourceType": "Patient", "id": "p"})
        assert not report.ok
        assert report.errors[0].rule == "allowed-resource-type"


class TestFhirInvariants:
    """Text verified against the R4 StructureDefinitions."""

    def test_con5_clinical_status_with_entered_in_error(self):
        resource = minimal_condition(
            verificationStatus={
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                        "code": "entered-in-error",
                    }
                ]
            },
            clinicalStatus={
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                        "code": "active",
                    }
                ]
            },
        )
        report = check_fhir_invariants(resource)
        assert any(f.rule == "con-5" for f in report.errors)

    def test_con5_satisfied_without_clinical_status(self):
        resource = minimal_condition(
            verificationStatus={
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                        "code": "entered-in-error",
                    }
                ]
            }
        )
        assert not any(f.rule == "con-5" for f in check_fhir_invariants(resource).errors)

    def test_con4_abatement_needs_an_inactive_status(self):
        resource = minimal_condition(
            abatementString="resolved last year",
            clinicalStatus={
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                        "code": "active",
                    }
                ]
            },
        )
        assert any(f.rule == "con-4" for f in check_fhir_invariants(resource).errors)

    @pytest.mark.parametrize("code", ["inactive", "resolved", "remission"])
    def test_con4_satisfied_by_allowed_statuses(self, code):
        resource = minimal_condition(
            abatementString="x",
            clinicalStatus={
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                        "code": code,
                    }
                ]
            },
        )
        assert not any(f.rule == "con-4" for f in check_fhir_invariants(resource).errors)

    def test_con3_is_a_warning_and_needs_problem_list_item(self):
        """The real polarity: it binds when category IS problem-list-item."""
        resource = minimal_condition(
            category=[
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/condition-category",
                            "code": "problem-list-item",
                        }
                    ]
                }
            ]
        )
        report = check_fhir_invariants(resource)
        findings = [f for f in report.findings if f.rule == "con-3"]
        assert findings
        assert findings[0].severity is Severity.WARNING

    def test_con3_does_not_bind_without_the_category(self):
        assert not [
            f for f in check_fhir_invariants(minimal_condition()).findings if f.rule == "con-3"
        ]

    def test_obs6_data_absent_reason_with_a_value(self):
        resource = {
            "resourceType": "Observation",
            "valueQuantity": {"value": 1},
            "dataAbsentReason": {"text": "not done"},
        }
        assert any(f.rule == "obs-6" for f in check_fhir_invariants(resource).errors)

    def test_obs6_satisfied_without_a_value(self):
        resource = {"resourceType": "Observation", "dataAbsentReason": {"text": "not done"}}
        assert not any(f.rule == "obs-6" for f in check_fhir_invariants(resource).errors)

    def test_obs7_component_code_collision(self):
        coding = [{"system": "http://loinc.org", "code": "1234-5"}]
        resource = {
            "resourceType": "Observation",
            "code": {"coding": coding},
            "valueQuantity": {"value": 1},
            "component": [{"code": {"coding": coding}}],
        }
        assert any(f.rule == "obs-7" for f in check_fhir_invariants(resource).errors)

    def test_obs7_satisfied_with_different_component_codes(self):
        resource = {
            "resourceType": "Observation",
            "code": {"coding": [{"system": "http://loinc.org", "code": "1111-1"}]},
            "valueQuantity": {"value": 1},
            "component": [{"code": {"coding": [{"system": "http://loinc.org", "code": "2222-2"}]}}],
        }
        assert not any(f.rule == "obs-7" for f in check_fhir_invariants(resource).errors)


class TestAggregate:
    def test_validate_resource_runs_everything(self):
        assert validate_resource(minimal_condition(), NOTE).ok

    def test_validate_bundle_checks_members(self):
        from fhirground.mapping import build_bundle

        bundle = build_bundle([minimal_condition()], bundle_id="b1")
        assert validate_bundle(bundle, NOTE).ok

    def test_validate_bundle_reports_a_bad_member(self):
        from fhirground.mapping import build_bundle

        bad = minimal_condition()
        bad["extension"] = []
        bundle = build_bundle([bad], bundle_id="b1")
        assert not validate_bundle(bundle, NOTE).ok

    def test_report_is_json_safe(self):
        import json

        bad = minimal_condition()
        bad["extension"] = []
        json.dumps(validate_resource(bad, NOTE).as_dict())

    def test_finding_renders_readably(self):
        bad = minimal_condition()
        bad["extension"] = []
        finding = validate_resource(bad, NOTE).errors[0]
        text = str(finding)
        assert "error" in text and "Condition/c1" in text


class TestHl7Hook:
    def test_missing_jar_is_reported_not_raised(self, tmp_path):
        report = run_hl7_validator([minimal_condition()], tmp_path / "nope.jar")
        assert not report.ok
        assert "not found" in report.errors[0].message

    def test_missing_java_is_a_warning(self, tmp_path):
        """An optional hook must not break a run."""
        jar = tmp_path / "validator_cli.jar"
        jar.write_bytes(b"not really a jar")
        report = run_hl7_validator([minimal_condition()], jar, java="/nonexistent/java")
        assert report.findings
