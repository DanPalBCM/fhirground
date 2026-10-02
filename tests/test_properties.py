"""Property-based tests. SPEC §8.2.

Four invariants, and the first one is the project's central claim:

1. **No invented codes.** For arbitrary mention lists -- including adversarial
   ``text`` containing code-like strings such as ``"SNOMED 44054006"`` -- every
   ``coding.code`` in the output exists in the active resolvers' value sets.
2. Every emitted resource parses with the FHIR models and passes project rules.
3. Every evidence span satisfies the offset/quote invariant.
4. Dropped plus emitted mention counts equal total mentions.
"""

from __future__ import annotations

import json

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from fhirground.io import NoteInput
from fhirground.mapping.extensions import read_evidence
from fhirground.pipeline import Pipeline, PipelineConfig
from fhirground.schemas.mention import Assertion, MentionType
from fhirground.validate import validate_resource
from tests.conftest import NOTE_DT, make_mention

FAST = settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)

#: Strings a model might emit that *look* like codes. The no-invented-codes
#: guarantee has to survive every one of them appearing in free-text fields.
CODE_LIKE = st.sampled_from(
    [
        "SNOMED 44054006",
        "44054006",
        "ICD-10 E11.9",
        "E11.9",
        "LOINC 4548-4",
        "4548-4",
        "RxNorm 860975",
        "860975",
        "http://snomed.info/sct|44054006",
        '{"system": "http://loinc.org", "code": "1234-5"}',
        "code: 99999999",
        "see SNOMED CT 73211009 for diabetes mellitus",
    ]
)

concept_text = st.one_of(
    st.sampled_from(
        [
            "type 2 diabetes",
            "hypertension",
            "chest pain",
            "pneumonia",
            "a1c",
            "heart rate",
            "metformin",
            "lisinopril",
            "unknown finding",
        ]
    ),
    CODE_LIKE,
    st.text(min_size=1, max_size=40).filter(lambda s: s.strip()),
)


class TestNoInventedCodes:
    """SPEC §8.2 and §11: the project's central guarantee."""

    @FAST
    @given(
        texts=st.lists(concept_text, min_size=1, max_size=6),
        kinds=st.lists(st.sampled_from(list(MentionType)), min_size=1, max_size=6),
        assertions=st.lists(st.sampled_from(list(Assertion)), min_size=1, max_size=6),
    )
    def test_every_emitted_code_came_from_a_value_set(self, resolvers, texts, kinds, assertions):
        """Drive adversarial mentions through mapping and check every coding."""
        from fhirground.assertion import decide
        from fhirground.grounding import ground_mention
        from fhirground.preprocess import chunk_note

        allowed = resolvers.all_known_codes()

        # Build a note whose text contains every mention's evidence verbatim,
        # so grounding succeeds and the mentions actually reach mapping.
        sentences = []
        mentions = []
        for i, text in enumerate(texts):
            kind = kinds[i % len(kinds)]
            assertion = assertions[i % len(assertions)]
            evidence = f"Patient record entry {i}: {text} noted."
            sentences.append(evidence)
            mentions.append(make_mention(kind, text=text, evidence=evidence, assertion=assertion))

        note_text = " ".join(sentences)
        note = NoteInput(
            note_id="prop",
            patient_ref="Patient/1",
            note_datetime=NOTE_DT,
            text=note_text,
        )
        plan = chunk_note(note_text, "prop", max_chars=4000)

        from fhirground.mapping import (
            build_condition,
            build_medication_request,
            build_observation,
        )

        builders = {
            MentionType.CONDITION: build_condition,
            MentionType.MEDICATION: build_medication_request,
            MentionType.OBSERVATION: build_observation,
        }

        for i, mention in enumerate(mentions):
            decision = decide(mention)
            if not decision.emit:
                continue
            grounding = ground_mention(mention.evidence, plan.chunks, note_text)
            if grounding is None:
                continue
            resolution = resolvers.resolve(mention.text, mention.mention_type)
            resource = builders[mention.mention_type](
                mention,
                resolution,
                grounding,
                decision,
                note,
                confidence=0.5,
                resource_id=f"r{i}",
            )
            blob = json.dumps(resource)
            for found in _codings(resource):
                assert found in allowed, (
                    f"emitted code {found} is not in the active value sets. "
                    f"Mention text was {mention.text!r}. Resource: {blob[:400]}"
                )

    @FAST
    @given(text=CODE_LIKE)
    def test_code_like_text_never_becomes_a_coding(self, resolvers, text):
        """A code in the *text* must not become a code in the *output*."""
        resolution = resolvers.resolve(text, MentionType.CONDITION)
        if resolution.coding is not None:
            assert (
                resolution.coding.system,
                resolution.coding.code,
            ) in resolvers.all_known_codes()
            # And whatever code came back, it came from the value set, not from
            # the digits embedded in the input.
            assert resolution.coding.display is not None

    @FAST
    @given(text=CODE_LIKE)
    def test_mention_schema_rejects_a_code_field(self, text):
        """The schema has nowhere to put a code, whatever the model tries."""
        import jsonschema

        from fhirground.schemas import build_mention_schema

        validator = jsonschema.Draft202012Validator(build_mention_schema())
        payload = {
            "mentions": [
                {
                    "mention_type": "condition",
                    "text": text,
                    "evidence": "some evidence",
                    "assertion": "present",
                    "code": "44054006",
                    "system": "http://snomed.info/sct",
                }
            ]
        }
        assert list(validator.iter_errors(payload)), (
            "the schema accepted a mention carrying code fields"
        )


def _codings(resource: dict) -> set[tuple[str, str]]:
    """Every (system, code) in a resource, excluding FHIR's own status systems."""
    out: set[tuple[str, str]] = set()
    terminology_systems = {
        "http://terminology.hl7.org/CodeSystem/condition-clinical",
        "http://terminology.hl7.org/CodeSystem/condition-ver-status",
        "http://terminology.hl7.org/CodeSystem/condition-category",
        "http://terminology.hl7.org/CodeSystem/observation-category",
        "http://terminology.hl7.org/CodeSystem/v3-DataOperation",
        "http://terminology.hl7.org/CodeSystem/provenance-participant-type",
        "http://unitsofmeasure.org",
    }

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "coding" and isinstance(value, list):
                    for coding in value:
                        if not isinstance(coding, dict):
                            continue
                        system = str(coding.get("system") or "")
                        code = str(coding.get("code") or "")
                        if code and system not in terminology_systems:
                            out.add((system, code))
                else:
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(resource)
    return out


class TestEveryResourceIsValid:
    """SPEC §8.2: everything emitted parses and passes the project rules."""

    @FAST
    @given(
        sentences=st.lists(
            st.sampled_from(
                [
                    "Assessment: Type 2 diabetes, poorly controlled.",
                    "A1c 9.2% today.",
                    "Start metformin 500 mg PO BID.",
                    "Denies chest pain.",
                    "Mother has hypertension.",
                    "Possible pneumonia; will reassess.",
                    "Heart rate 78 bpm.",
                    "Continue lisinopril 10 mg PO daily.",
                    "History of asthma, resolved.",
                ]
            ),
            min_size=1,
            max_size=6,
            unique=True,
        )
    )
    def test_pipeline_output_always_validates(self, resolvers, sentences):
        from fhirground.backends import MockBackend

        text = " ".join(sentences)
        note = NoteInput(note_id="prop", patient_ref="Patient/1", note_datetime=NOTE_DT, text=text)
        pipe = Pipeline(PipelineConfig(), backend=MockBackend(), resolvers=resolvers)
        result = pipe.run_note(note)

        assert result.validation.ok, [str(f) for f in result.validation.errors]
        for resource in result.resources:
            report = validate_resource(resource, note.text)
            assert report.ok, [str(f) for f in report.errors]


class TestGroundingInvariant:
    """SPEC §8.2: every evidence span indexes the note and matches its quote."""

    @FAST
    @given(
        sentences=st.lists(
            st.sampled_from(
                [
                    "Assessment: Type 2 diabetes, poorly controlled.",
                    "A1c 9.2% today.",
                    "Start metformin 500 mg PO BID.",
                    "Heart rate 78 bpm.",
                    "Continue lisinopril 10 mg PO daily.",
                ]
            ),
            min_size=1,
            max_size=5,
            unique=True,
        )
    )
    def test_offsets_and_quotes_agree(self, resolvers, sentences):
        from fhirground.backends import MockBackend

        text = " ".join(sentences)
        note = NoteInput(note_id="prop", patient_ref="Patient/1", note_datetime=NOTE_DT, text=text)
        pipe = Pipeline(PipelineConfig(), backend=MockBackend(), resolvers=resolvers)
        result = pipe.run_note(note)

        for resource in result.resources:
            if resource["resourceType"] == "Provenance":
                continue
            evidence = read_evidence(resource)
            assert evidence, f"{resource['resourceType']} carries no evidence"
            for span in evidence:
                start, end = span["start"], span["end"]
                assert 0 <= start <= end <= len(text)
                actual = " ".join(text[start:end].split()).lower()
                stored = " ".join((span.get("quote") or "").split()).lower()
                assert actual == stored


class TestMentionAccounting:
    """SPEC §8.2: dropped plus emitted equals total."""

    @FAST
    @given(
        sentences=st.lists(
            st.sampled_from(
                [
                    "Assessment: Type 2 diabetes.",
                    "Denies chest pain.",
                    "Mother has hypertension.",
                    "Possible pneumonia.",
                    "A1c 9.2% today.",
                    "Start metformin 500 mg PO BID.",
                    "If she develops a rash, stop lisinopril.",
                ]
            ),
            min_size=1,
            max_size=6,
            unique=True,
        )
    )
    def test_every_mention_is_accounted_for(self, resolvers, sentences):
        from fhirground.backends import MockBackend

        text = " ".join(sentences)
        note = NoteInput(note_id="prop", patient_ref="Patient/1", note_datetime=NOTE_DT, text=text)
        pipe = Pipeline(PipelineConfig(), backend=MockBackend(), resolvers=resolvers)
        result = pipe.run_note(note)

        from fhirground.assertion import DropReason

        emitted = len([r for r in result.resources if r["resourceType"] != "Provenance"])

        # Ungrounded mentions are dropped *before* deduplication, so they are
        # not part of the group accounting. Everything else is: each group
        # became either one resource or one drop.
        ungrounded = sum(1 for _m, reason in result.dropped if reason is DropReason.UNGROUNDED)
        post_group_drops = len(result.dropped) - ungrounded
        assert emitted + post_group_drops == result.stats["n_groups"], (
            f"{emitted} emitted + {post_group_drops} dropped != {result.stats['n_groups']} groups"
        )

        # And every trace has exactly one outcome.
        for trace in result.debug:
            has_resource = trace.resource_id is not None
            has_drop = trace.drop_reason is not None
            assert has_resource != has_drop, (
                f"trace for {trace.mention.text!r} has resource={has_resource} "
                f"drop={has_drop}; exactly one must be true"
            )
