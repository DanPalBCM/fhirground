"""Backends: request builders, the repair loop, and mock determinism.

SPEC §0 requires a test that the mention schema is accepted by each backend's
request builder. No test here reaches a network or AWS: the OpenAI-compatible
backend is driven with respx and Bedrock with botocore's Stubber.
"""

from __future__ import annotations

import json
from typing import ClassVar

import httpx
import pytest
import respx
from botocore.stub import Stubber

from fhirground.backends import (
    TOOL_NAME,
    BedrockBackend,
    MockBackend,
    OpenAICompatible,
    extract_by_rules,
    record_fixtures,
    render,
    render_repair,
)
from fhirground.backends.prompts import PROMPT_VERSION
from fhirground.exceptions import BackendError, SchemaValidationError
from fhirground.extract import CallCache, extract_chunk, parse_mentions, strip_to_json
from fhirground.preprocess import chunk_note
from fhirground.schemas import build_mention_schema

SCHEMA = build_mention_schema()
NOTE = "Assessment: Type 2 diabetes. Start metformin 500 mg PO BID."


class TestPrompts:
    def test_version_is_recorded(self):
        assert render("text").version == PROMPT_VERSION

    def test_chunk_is_delimited(self):
        prompt = render("some chunk text")
        assert "--- NOTE TEXT START ---" in prompt.user
        assert "some chunk text" in prompt.user

    def test_context_included_when_given(self):
        prompt = render("x", section="family history", note_datetime="2026-03-14")
        assert "family history" in prompt.user
        assert "2026-03-14" in prompt.user

    def test_system_prompt_forbids_codes(self):
        assert "Never output a medical code" in render("x").system

    def test_repair_prompt_carries_the_error(self):
        base = render("x")
        repair = render_repair(base, "field 'text': too long")
        assert "too long" in repair.user
        assert repair.version == base.version

    def test_digest_is_stable(self):
        assert render("x").digest() == render("x").digest()
        assert render("x").digest() != render("y").digest()


class TestOpenAICompatibleRequestBuilder:
    """SPEC §0: the schema must be embedded correctly."""

    @pytest.fixture
    def backend(self):
        return OpenAICompatible(base_url="http://localhost:8000/v1", model="m")

    def test_schema_embedded_verbatim(self, backend):
        body = backend.build_request("prompt", SCHEMA)
        assert body["response_format"]["type"] == "json_schema"
        assert body["response_format"]["json_schema"]["schema"] == SCHEMA

    def test_json_schema_keys_match_the_openai_contract(self, backend):
        js = backend.build_request("p", SCHEMA)["response_format"]["json_schema"]
        assert set(js) == {"name", "schema", "strict"}

    def test_strict_is_off_by_default(self, backend):
        """Strict mode would require every optional field to be present."""
        assert (
            backend.build_request("p", SCHEMA)["response_format"]["json_schema"]["strict"] is False
        )

    def test_strict_can_be_enabled(self):
        b = OpenAICompatible(base_url="http://x/v1", model="m", strict=True)
        assert b.build_request("p", SCHEMA)["response_format"]["json_schema"]["strict"] is True

    def test_guided_json_mode(self):
        b = OpenAICompatible(base_url="http://x/v1", model="m", constraint_mode="guided_json")
        body = b.build_request("p", SCHEMA)
        assert body["guided_json"] == SCHEMA
        assert "response_format" not in body

    def test_no_constraint_mode(self):
        b = OpenAICompatible(base_url="http://x/v1", model="m", constraint_mode="none")
        body = b.build_request("p", SCHEMA)
        assert "response_format" not in body
        assert "guided_json" not in body

    def test_unknown_constraint_mode_rejected(self):
        with pytest.raises(BackendError, match="unknown constraint_mode"):
            OpenAICompatible(base_url="http://x/v1", model="m", constraint_mode="magic")

    def test_system_message_included(self, backend):
        body = backend.build_request("user text", SCHEMA, system="sys text")
        assert body["messages"][0] == {"role": "system", "content": "sys text"}
        assert body["messages"][1]["content"] == "user text"

    def test_n_only_set_above_one(self, backend):
        assert "n" not in backend.build_request("p", SCHEMA, n=1)
        assert backend.build_request("p", SCHEMA, n=3)["n"] == 3


class TestOpenAICompatibleCalls:
    PAYLOAD: ClassVar[dict] = {
        "model": "m",
        "choices": [
            {
                "message": {"content": '{"mentions": []}'},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    }

    @respx.mock
    def test_successful_call(self):
        respx.post("http://localhost:8000/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=self.PAYLOAD)
        )
        b = OpenAICompatible(base_url="http://localhost:8000/v1", model="m")
        responses = b.extract("p", SCHEMA)
        assert len(responses) == 1
        assert responses[0].tokens_in == 100
        assert responses[0].latency_s is not None

    @respx.mock
    def test_auth_failure_is_a_backend_error(self):
        respx.post("http://localhost:8000/v1/chat/completions").mock(
            return_value=httpx.Response(401)
        )
        b = OpenAICompatible(base_url="http://localhost:8000/v1", model="m")
        with pytest.raises(BackendError, match="credentials"):
            b.extract("p", SCHEMA)

    @respx.mock
    def test_404_suggests_the_api_prefix(self):
        respx.post("http://localhost:8000/chat/completions").mock(return_value=httpx.Response(404))
        b = OpenAICompatible(base_url="http://localhost:8000", model="m")
        with pytest.raises(BackendError, match="/v1"):
            b.extract("p", SCHEMA)

    @respx.mock
    def test_connection_error_mentions_the_server(self):
        respx.post("http://localhost:8000/v1/chat/completions").mock(
            side_effect=httpx.ConnectError("refused")
        )
        b = OpenAICompatible(base_url="http://localhost:8000/v1", model="m")
        with pytest.raises(BackendError, match="Is the server running"):
            b.extract("p", SCHEMA)

    @respx.mock
    def test_no_choices(self):
        respx.post("http://localhost:8000/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": []})
        )
        b = OpenAICompatible(base_url="http://localhost:8000/v1", model="m")
        with pytest.raises(BackendError, match="no choices"):
            b.extract("p", SCHEMA)

    @respx.mock
    def test_logprobs_become_a_probability(self):
        payload = dict(self.PAYLOAD)
        payload["choices"] = [
            {
                "message": {"content": '{"mentions": []}'},
                "finish_reason": "stop",
                "logprobs": {"content": [{"logprob": -0.1}, {"logprob": -0.3}]},
            }
        ]
        respx.post("http://localhost:8000/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=payload)
        )
        b = OpenAICompatible(base_url="http://localhost:8000/v1", model="m")
        assert 0.0 < b.extract("p", SCHEMA)[0].logprob < 1.0

    def test_describe(self):
        d = OpenAICompatible(base_url="http://x/v1", model="m").describe()
        assert d["backend"] == "openai_compatible"
        assert d["constraint_mode"] == "json_schema"

    @pytest.mark.live_vllm
    def test_live_server(self):
        import os

        b = OpenAICompatible(
            base_url=os.environ["FHIRGROUND_LIVE_VLLM"],
            model=os.environ.get("FHIRGROUND_LIVE_VLLM_MODEL", "m"),
        )
        responses = b.extract(render(NOTE).combined, SCHEMA)
        assert parse_mentions(responses[0].text) is not None


class TestBedrockRequestBuilder:
    """SPEC §0: verified against botocore's own service model."""

    @pytest.fixture
    def backend(self):
        return BedrockBackend(model_id="anthropic.claude-x", region="us-east-1")

    def test_schema_embedded_in_the_tool(self, backend):
        request = backend.build_request("p", SCHEMA)
        tool_spec = request["toolConfig"]["tools"][0]["toolSpec"]
        assert tool_spec["inputSchema"]["json"] == SCHEMA

    def test_tool_is_forced(self, backend):
        request = backend.build_request("p", SCHEMA)
        assert request["toolConfig"]["toolChoice"] == {"tool": {"name": TOOL_NAME}}

    def test_tool_spec_required_keys(self, backend):
        tool_spec = backend.build_request("p", SCHEMA)["toolConfig"]["tools"][0]["toolSpec"]
        assert {"name", "inputSchema"} <= set(tool_spec)

    def test_strict_omitted_by_default(self, backend):
        tool_spec = backend.build_request("p", SCHEMA)["toolConfig"]["tools"][0]["toolSpec"]
        assert "strict" not in tool_spec

    def test_strict_can_be_enabled(self):
        b = BedrockBackend(model_id="m", strict=True)
        tool_spec = b.build_request("p", SCHEMA)["toolConfig"]["tools"][0]["toolSpec"]
        assert tool_spec["strict"] is True

    def test_system_message(self, backend):
        request = backend.build_request("p", SCHEMA, system="sys")
        assert request["system"] == [{"text": "sys"}]

    def test_inference_config(self, backend):
        request = backend.build_request("p", SCHEMA, temperature=0.4)
        assert request["inferenceConfig"]["temperature"] == 0.4

    def test_request_is_accepted_by_botocore(self, backend):
        """The strongest available offline check: botocore validates the shape."""
        import boto3

        client = boto3.client(
            "bedrock-runtime",
            region_name="us-east-1",
            aws_access_key_id="x",
            aws_secret_access_key="y",
        )
        request = backend.build_request("p", SCHEMA)
        response = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "t1",
                                "name": TOOL_NAME,
                                "input": {"mentions": []},
                            }
                        }
                    ],
                }
            },
            "stopReason": "tool_use",
            "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
            "metrics": {"latencyMs": 100},
        }
        with Stubber(client) as stubber:
            stubber.add_response("converse", response, request)
            backend._client = client
            out = backend.extract("p", SCHEMA)
        assert json.loads(out[0].text) == {"mentions": []}
        assert out[0].tokens_in == 10

    def test_prose_response_passes_through_for_repair(self, backend):
        parsed = backend._parse(
            {
                "output": {"message": {"content": [{"text": "I cannot do that."}]}},
                "stopReason": "end_turn",
                "usage": {},
            },
            0.1,
        )
        assert parsed.text == "I cannot do that."

    @pytest.mark.parametrize(
        ("code", "pattern"),
        [
            ("AccessDeniedException", "credentials"),
            ("ThrottlingException", "throttled"),
            ("ValidationException", "rejected the request"),
            ("ResourceNotFoundException", "no model"),
        ],
    )
    def test_error_translation(self, backend, code, pattern):
        exc = Exception("boom")
        exc.response = {"Error": {"Code": code}}
        assert pattern in str(backend._translate(exc))

    def test_missing_boto3_is_explained(self, monkeypatch):
        import builtins

        real = builtins.__import__

        def fake(name, *args, **kwargs):
            if name == "boto3":
                raise ImportError("no boto3")
            return real(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake)
        b = BedrockBackend(model_id="m")
        with pytest.raises(BackendError, match=r"fhirground\[bedrock\]"):
            b.extract("p", SCHEMA)

    def test_describe(self, backend):
        assert backend.describe()["backend"] == "bedrock"


class TestMockBackend:
    def test_deterministic(self):
        a, b = MockBackend(), MockBackend()
        prompt = render(NOTE).combined
        assert a.extract(prompt, SCHEMA)[0].text == b.extract(prompt, SCHEMA)[0].text

    def test_rule_extraction_finds_mentions(self):
        mentions = extract_by_rules(NOTE)
        kinds = {m["mention_type"] for m in mentions}
        assert "condition" in kinds
        assert "medication" in kinds

    def test_rules_detect_negation(self):
        mentions = extract_by_rules("Denies chest pain.")
        assert mentions[0]["assertion"] == "negated"

    def test_rules_detect_family_history(self):
        mentions = extract_by_rules("Mother has hypertension.")
        assert mentions[0]["assertion"] == "family"

    def test_rules_detect_hedging(self):
        mentions = extract_by_rules("Possible pneumonia.")
        assert mentions[0]["assertion"] == "possible"

    def test_rules_detect_history(self):
        mentions = extract_by_rules("History of asthma.")
        assert mentions[0]["assertion"] == "historical"

    def test_rules_detect_hypothetical(self):
        mentions = extract_by_rules("If he develops asthma, start albuterol.")
        assert mentions[0]["assertion"] == "hypothetical"

    def test_rules_extract_dose_and_route(self):
        mentions = extract_by_rules("Start metformin 500 mg PO BID.")
        med = next(m for m in mentions if m["mention_type"] == "medication")
        assert med["dose_value"] == 500.0
        assert med["dose_unit"] == "mg"
        assert med["route"] == "PO"
        assert med["frequency_text"] == "BID"
        assert med["med_action"] == "start"

    def test_rules_never_emit_a_code_field(self):
        """Even given a note full of codes."""
        mentions = extract_by_rules("Diabetes SNOMED 44054006 ICD-10 E11.9 LOINC 4548-4.")
        for mention in mentions:
            assert not any("code" in k for k in mention)

    def test_fixtures_take_priority(self):
        chunk = NOTE
        fixtures = {
            MockBackend.fixture_key(chunk): [
                {"mention_type": "condition", "text": "x", "evidence": "y", "assertion": "present"}
            ]
        }
        backend = MockBackend(fixtures=fixtures)
        payload = json.loads(backend.extract(render(chunk).combined, SCHEMA)[0].text)
        assert payload["mentions"][0]["text"] == "x"

    def test_rules_can_be_disabled(self):
        backend = MockBackend(rules=False)
        payload = json.loads(backend.extract(render(NOTE).combined, SCHEMA)[0].text)
        assert payload["mentions"] == []

    def test_fail_first_n_returns_malformed_json(self):
        backend = MockBackend(fail_first_n=1)
        first = backend.extract(render(NOTE).combined, SCHEMA)[0].text
        with pytest.raises(SchemaValidationError):
            parse_mentions(first)
        second = backend.extract(render(NOTE).combined, SCHEMA)[0].text
        assert parse_mentions(second) is not None

    def test_fixture_file_round_trip(self, tmp_path):
        path = tmp_path / "f.json"
        path.write_text(json.dumps(record_fixtures([NOTE])))
        backend = MockBackend.from_fixture_file(path)
        assert len(backend.fixtures) == 1

    def test_bad_fixture_file(self, tmp_path):
        path = tmp_path / "f.json"
        path.write_text("[]")
        with pytest.raises(BackendError, match="object of chunk"):
            MockBackend.from_fixture_file(path)

    def test_missing_fixture_file(self, tmp_path):
        with pytest.raises(BackendError, match="not found"):
            MockBackend.from_fixture_file(tmp_path / "nope.json")

    def test_describe(self):
        assert MockBackend().describe()["backend"] == "mock"


class TestParsing:
    def test_plain_json(self):
        assert len(parse_mentions('{"mentions": []}')) == 0

    def test_markdown_fence_stripped(self):
        assert len(parse_mentions('```json\n{"mentions": []}\n```')) == 0

    def test_surrounding_prose_stripped(self):
        text = 'Here you go:\n{"mentions": []}\nHope that helps!'
        assert len(parse_mentions(text)) == 0

    def test_bare_array_accepted(self):
        payload = (
            '[{"mention_type": "condition", "text": "t", "evidence": "e", "assertion": "present"}]'
        )
        assert len(parse_mentions(payload)) == 1

    def test_empty_response(self):
        with pytest.raises(SchemaValidationError, match="empty"):
            parse_mentions("")

    def test_malformed_json_names_the_position(self):
        with pytest.raises(SchemaValidationError, match="not valid JSON"):
            parse_mentions('{"mentions": [')

    def test_non_object_top_level(self):
        with pytest.raises(SchemaValidationError, match="must be a JSON object"):
            parse_mentions('"a string"')

    def test_extra_field_error_mentions_codes(self):
        """The repair message has to tell the model what it did wrong."""
        payload = json.dumps(
            {
                "mentions": [
                    {
                        "mention_type": "condition",
                        "text": "t",
                        "evidence": "e",
                        "assertion": "present",
                        "snomed_code": "1",
                    }
                ]
            }
        )
        with pytest.raises(SchemaValidationError, match="no code fields"):
            parse_mentions(payload)

    def test_strip_to_json_helper(self):
        assert strip_to_json("```json\n{}\n```") == "{}"
        assert strip_to_json("noise {} noise") == "{}"
        assert strip_to_json("") == ""


class TestRepairLoop:
    def test_succeeds_on_the_first_try(self):
        chunk = chunk_note(NOTE, "n1").chunks[0]
        result = extract_chunk(chunk, MockBackend(), SCHEMA)
        assert result.ok
        assert result.repairs == 0

    def test_repairs_once(self):
        chunk = chunk_note(NOTE, "n1").chunks[0]
        result = extract_chunk(chunk, MockBackend(fail_first_n=1), SCHEMA, max_repairs=2)
        assert result.ok
        assert result.repairs == 1

    def test_gives_up_after_max_repairs(self):
        chunk = chunk_note(NOTE, "n1").chunks[0]
        result = extract_chunk(chunk, MockBackend(fail_first_n=99), SCHEMA, max_repairs=2)
        assert not result.ok
        assert result.repairs == 2
        assert "not valid JSON" in result.error

    def test_zero_repairs_allowed(self):
        chunk = chunk_note(NOTE, "n1").chunks[0]
        result = extract_chunk(chunk, MockBackend(fail_first_n=1), SCHEMA, max_repairs=0)
        assert not result.ok

    def test_token_and_latency_accumulate(self):
        chunk = chunk_note(NOTE, "n1").chunks[0]
        result = extract_chunk(chunk, MockBackend(latency_s=0.5), SCHEMA)
        assert result.tokens_in > 0
        assert result.latency_s == pytest.approx(0.5)


class TestCallCache:
    def test_hit_on_the_second_call(self, tmp_path):
        cache = CallCache(tmp_path / "c.sqlite")
        chunk = chunk_note(NOTE, "n1").chunks[0]
        backend = MockBackend()
        extract_chunk(chunk, backend, SCHEMA, cache=cache, schema_digest="d")
        calls_after_first = backend.calls
        extract_chunk(chunk, backend, SCHEMA, cache=cache, schema_digest="d")
        assert backend.calls == calls_after_first, "the second call should not reach the backend"
        assert cache.stats()["hits"] >= 1

    def test_key_changes_with_the_prompt(self):
        a = CallCache.make_key({}, "p1", temperature=0.0, n=1, schema_digest="d")
        b = CallCache.make_key({}, "p2", temperature=0.0, n=1, schema_digest="d")
        assert a != b

    def test_key_changes_with_the_schema(self):
        a = CallCache.make_key({}, "p", temperature=0.0, n=1, schema_digest="d1")
        b = CallCache.make_key({}, "p", temperature=0.0, n=1, schema_digest="d2")
        assert a != b

    def test_cached_responses_are_marked(self, tmp_path):
        cache = CallCache(tmp_path / "c.sqlite")
        from fhirground.backends.base import RawResponse

        cache.put("k", [RawResponse(text="{}", model_id="m")])
        assert cache.get("k")[0].cached is True
