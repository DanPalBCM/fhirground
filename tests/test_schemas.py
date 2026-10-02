"""The mention schema: the §0 hard rule, and the backend-safe subset."""

from __future__ import annotations

import json

import jsonschema
import pytest
from pydantic import ValidationError

from fhirground.schemas import (
    MAX_EVIDENCE_LEN,
    MAX_TEXT_LEN,
    SCHEMA_PATH,
    Assertion,
    Mention,
    MentionList,
    MentionType,
    assert_backend_safe,
    assert_no_code_fields,
    build_mention_schema,
    load_mention_schema,
    schema_digest,
    write_mention_schema,
)
from fhirground.schemas.mention import FORBIDDEN_FIELD_SUBSTRINGS


class TestNoCodeFields:
    """SPEC §0: the LLM never emits medical codes."""

    def test_schema_has_no_code_field(self):
        assert_no_code_fields()

    def test_the_guard_actually_catches_one(self):
        """The guard must fail on a code-shaped name, or it proves nothing."""
        for bad in ("snomed_code", "coding", "icd10", "concept_id", "code"):
            assert any(s in bad for s in FORBIDDEN_FIELD_SUBSTRINGS), bad

    def test_extra_fields_are_rejected_by_the_model(self):
        with pytest.raises(ValidationError):
            Mention(
                mention_type=MentionType.CONDITION,
                text="t",
                evidence="e",
                assertion=Assertion.PRESENT,
                snomed_code="44054006",
            )

    def test_generated_schema_forbids_additional_properties(self):
        schema = build_mention_schema()
        assert schema["additionalProperties"] is False
        assert schema["properties"]["mentions"]["items"]["additionalProperties"] is False


class TestBackendSafety:
    """SPEC §6.1: the subset Bedrock and vLLM both accept."""

    def test_generated_schema_is_backend_safe(self):
        assert_backend_safe(build_mention_schema())

    def test_no_refs_or_defs(self):
        blob = json.dumps(build_mention_schema())
        assert "$ref" not in blob
        assert "$defs" not in blob

    def test_no_anyof(self):
        assert "anyOf" not in json.dumps(build_mention_schema())

    @pytest.mark.parametrize("bad", ["$ref", "oneOf", "allOf", "anyOf", "not"])
    def test_guard_rejects_forbidden_constructs(self, bad):
        with pytest.raises(ValueError, match=bad.replace("$", r"\$")):
            assert_backend_safe({"type": "object", "properties": {"x": {bad: "y"}}})

    def test_guard_reports_the_path(self):
        with pytest.raises(ValueError, match=r"properties\.x"):
            assert_backend_safe({"properties": {"x": {"$ref": "#/$defs/Y"}}})

    def test_schema_is_valid_draft_2020_12(self):
        jsonschema.Draft202012Validator.check_schema(build_mention_schema())


class TestNullableFields:
    """The §6.1 typo fix: nullable fields use type unions, not enum literals."""

    def test_onset_text_is_a_nullable_string(self):
        props = build_mention_schema()["properties"]["mentions"]["items"]["properties"]
        assert props["onset_text"]["type"] == ["string", "null"]

    def test_nullable_enums_allow_null_in_their_declared_type(self):
        props = build_mention_schema()["properties"]["mentions"]["items"]["properties"]
        for name in ("clinical_status_hint", "med_action", "obs_category", "value_comparator"):
            assert None in props[name]["enum"], name
            assert "null" in props[name]["type"], name

    def test_a_validator_accepts_explicit_nulls(self):
        validator = jsonschema.Draft202012Validator(build_mention_schema())
        payload = {
            "mentions": [
                {
                    "mention_type": "condition",
                    "text": "t",
                    "evidence": "e",
                    "assertion": "present",
                    "clinical_status_hint": None,
                    "onset_text": None,
                    "med_action": None,
                    "dose_value": None,
                    "obs_category": None,
                    "value_comparator": None,
                }
            ]
        }
        assert not list(validator.iter_errors(payload))


class TestRequiredFields:
    def test_exactly_four_required(self):
        items = build_mention_schema()["properties"]["mentions"]["items"]
        assert items["required"] == ["mention_type", "text", "evidence", "assertion"]

    def test_missing_a_required_field_is_rejected(self):
        validator = jsonschema.Draft202012Validator(build_mention_schema())
        bad = {"mentions": [{"mention_type": "condition", "text": "t"}]}
        assert list(validator.iter_errors(bad))


class TestCheckedInFile:
    def test_file_matches_the_model(self):
        """The checked-in schema must not drift from the Pydantic source."""
        assert SCHEMA_PATH.exists(), "run `fhirground schema --out` to generate it"
        assert json.loads(SCHEMA_PATH.read_text()) == build_mention_schema()

    def test_load_returns_the_file(self):
        assert load_mention_schema() == build_mention_schema()

    def test_write_round_trips(self, tmp_path):
        path = write_mention_schema(tmp_path / "s.json")
        assert json.loads(path.read_text()) == build_mention_schema()

    def test_digest_is_stable_and_short(self):
        assert schema_digest() == schema_digest(build_mention_schema())
        assert len(schema_digest()) == 16


class TestMentionModel:
    def test_blank_optional_strings_become_none(self):
        """Models emit "" where they mean null; carrying it through would
        produce an empty route.text in the FHIR output."""
        m = Mention(
            mention_type=MentionType.MEDICATION,
            text="metformin",
            evidence="start metformin",
            assertion=Assertion.PRESENT,
            route="   ",
            frequency_text="",
            onset_text="  ",
        )
        assert m.route is None
        assert m.frequency_text is None
        assert m.onset_text is None

    def test_text_and_evidence_are_stripped(self):
        m = Mention(
            mention_type=MentionType.CONDITION,
            text="  diabetes  ",
            evidence="  Diabetes noted.  ",
            assertion=Assertion.PRESENT,
        )
        assert m.text == "diabetes"
        assert m.evidence == "Diabetes noted."

    def test_empty_text_rejected(self):
        with pytest.raises(ValidationError):
            Mention(
                mention_type=MentionType.CONDITION,
                text="",
                evidence="e",
                assertion=Assertion.PRESENT,
            )

    def test_length_caps_enforced(self):
        with pytest.raises(ValidationError):
            Mention(
                mention_type=MentionType.CONDITION,
                text="x" * (MAX_TEXT_LEN + 1),
                evidence="e",
                assertion=Assertion.PRESENT,
            )
        with pytest.raises(ValidationError):
            Mention(
                mention_type=MentionType.CONDITION,
                text="t",
                evidence="x" * (MAX_EVIDENCE_LEN + 1),
                assertion=Assertion.PRESENT,
            )

    def test_normalized_text_collapses_case_and_space(self):
        m = Mention(
            mention_type=MentionType.CONDITION,
            text="Type  2   DIABETES",
            evidence="e",
            assertion=Assertion.PRESENT,
        )
        assert m.normalized_text == "type 2 diabetes"

    def test_mention_list_helpers(self):
        ml = MentionList(
            mentions=[
                Mention(
                    mention_type=MentionType.CONDITION,
                    text="a",
                    evidence="e",
                    assertion=Assertion.PRESENT,
                ),
                Mention(
                    mention_type=MentionType.MEDICATION,
                    text="b",
                    evidence="e",
                    assertion=Assertion.PRESENT,
                ),
            ]
        )
        assert len(ml) == 2
        assert len(ml.by_type(MentionType.CONDITION)) == 1
        assert [m.text for m in ml] == ["a", "b"]

    def test_empty_mention_list_is_valid(self):
        assert len(MentionList()) == 0
