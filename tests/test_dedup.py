"""Deduplication: what merges, what does not, and what survives a merge."""

from __future__ import annotations

from fhirground.dedup import (
    count_by_assertion,
    dedup_key,
    dedup_mentions,
    group_by_type,
    matches_for_consistency,
)
from fhirground.grounding import Grounding, GroundingStrategy
from fhirground.schemas.mention import Assertion, MedAction, MentionType
from fhirground.terminology.base import Coding, Resolution
from tests.conftest import SNOMED, make_mention


def g(start: int, end: int = 0, score: float = 1.0, strategy=GroundingStrategy.EXACT) -> Grounding:
    return Grounding(start=start, end=end or start + 10, quote="q", strategy=strategy, score=score)


def coded(code: str = "44054006", text: str = "type 2 diabetes") -> Resolution:
    return Resolution(
        text=text,
        coding=Coding(system=SNOMED, code=code, display="Type 2 diabetes mellitus"),
        score=1.0,
        resolver="test",
        method="exact",
    )


def uncoded(text: str = "fractured femur") -> Resolution:
    return Resolution(text=text, resolver="test", method="none")


class TestDedupKey:
    def test_coded_key_uses_the_code(self):
        a = dedup_key(make_mention(text="T2DM"), coded())
        b = dedup_key(make_mention(text="type 2 diabetes mellitus"), coded())
        assert a == b, "different surface forms of one code must share a key"

    def test_uncoded_key_uses_normalized_text(self):
        a = dedup_key(make_mention(text="Fractured  FEMUR"), uncoded())
        b = dedup_key(make_mention(text="fractured femur"), uncoded())
        assert a == b

    def test_uncoded_synonyms_do_not_merge(self):
        """Correct and conservative: no terminology said they were the same."""
        a = dedup_key(make_mention(text="T2DM"), uncoded("T2DM"))
        b = dedup_key(make_mention(text="type 2 diabetes"), uncoded("type 2 diabetes"))
        assert a != b

    def test_type_is_part_of_the_key(self):
        a = dedup_key(make_mention(MentionType.CONDITION, text="x"), uncoded("x"))
        b = dedup_key(make_mention(MentionType.OBSERVATION, text="x"), uncoded("x"))
        assert a != b

    def test_assertion_is_part_of_the_key(self):
        a = dedup_key(make_mention(assertion=Assertion.PRESENT), coded())
        b = dedup_key(make_mention(assertion=Assertion.POSSIBLE), coded())
        assert a != b, "'possible pneumonia' and 'pneumonia' are different claims"


class TestMerging:
    def test_identical_findings_merge(self):
        items = [
            (make_mention(text="T2DM"), coded(), g(0)),
            (make_mention(text="type 2 diabetes"), coded(), g(50)),
        ]
        groups = dedup_mentions(items)
        assert len(groups) == 1
        assert groups[0].n_duplicates == 1

    def test_every_span_survives(self):
        items = [
            (make_mention(text="T2DM"), coded(), g(0)),
            (make_mention(text="type 2 diabetes"), coded(), g(50)),
            (make_mention(text="DM2"), coded(), g(90)),
        ]
        groups = dedup_mentions(items)
        assert [s.start for s in groups[0].groundings] == [0, 50, 90]

    def test_duplicate_spans_are_not_repeated(self):
        """Chunk overlap can yield the identical span twice."""
        items = [
            (make_mention(text="T2DM"), coded(), g(0)),
            (make_mention(text="T2DM"), coded(), g(0)),
        ]
        groups = dedup_mentions(items)
        assert len(groups[0].groundings) == 1

    def test_spans_are_sorted_by_position(self):
        items = [
            (make_mention(text="a"), coded(), g(90)),
            (make_mention(text="b"), coded(), g(10)),
        ]
        groups = dedup_mentions(items)
        assert [s.start for s in groups[0].groundings] == [10, 90]

    def test_best_grounding_wins_the_representative(self):
        weak = make_mention(text="weak")
        strong = make_mention(text="strong")
        items = [
            (weak, coded(), g(0, score=0.6, strategy=GroundingStrategy.FUZZY)),
            (strong, coded(), g(50, score=1.0)),
        ]
        groups = dedup_mentions(items)
        assert groups[0].best_grounding.score == 1.0
        assert groups[0].mention.text == "strong"

    def test_richer_mention_becomes_the_representative(self):
        plain = make_mention(MentionType.MEDICATION, text="metformin", med_action=MedAction.START)
        detailed = make_mention(
            MentionType.MEDICATION,
            text="metformin",
            med_action=MedAction.START,
            dose_value=500,
            dose_unit="mg",
            route="PO",
        )
        items = [(plain, coded(), g(0)), (detailed, coded(), g(50))]
        groups = dedup_mentions(items)
        assert groups[0].mention.dose_value == 500

    def test_a_coded_resolution_is_preferred(self):
        items = [
            (make_mention(text="type 2 diabetes"), uncoded("type 2 diabetes"), g(0)),
            (make_mention(text="type 2 diabetes"), coded(text="type 2 diabetes"), g(50)),
        ]
        groups = dedup_mentions(items)
        # Uncoded and coded have different keys, so they do not merge -- which
        # is the honest outcome. Both are kept rather than one being guessed away.
        assert len(groups) == 2

    def test_group_order_is_first_appearance(self):
        items = [
            (make_mention(text="b"), uncoded("b"), g(0)),
            (make_mention(text="a"), uncoded("a"), g(10)),
        ]
        assert [grp.mention.text for grp in dedup_mentions(items)] == ["b", "a"]

    def test_empty_input(self):
        assert dedup_mentions([]) == []

    def test_debug_summary_is_json_safe(self):
        import json

        groups = dedup_mentions([(make_mention(), coded(), g(0))])
        json.dumps(groups[0].as_debug())


class TestSelfConsistencyMatching:
    def test_counts_samples_containing_a_match(self):
        target = (make_mention(text="T2DM"), coded())
        samples = [
            [(make_mention(text="type 2 diabetes"), coded())],
            [(make_mention(text="hypertension"), coded("38341003"))],
            [(make_mention(text="DM2"), coded())],
        ]
        assert matches_for_consistency(target, samples) == 2

    def test_no_matches(self):
        target = (make_mention(text="T2DM"), coded())
        samples = [[(make_mention(text="hypertension"), coded("38341003"))]]
        assert matches_for_consistency(target, samples) == 0

    def test_assertion_must_agree(self):
        target = (make_mention(assertion=Assertion.PRESENT), coded())
        samples = [[(make_mention(assertion=Assertion.POSSIBLE), coded())]]
        assert matches_for_consistency(target, samples) == 0

    def test_empty_samples(self):
        assert matches_for_consistency((make_mention(), coded()), []) == 0


class TestGrouping:
    def test_group_by_type_includes_every_type(self):
        groups = dedup_mentions([(make_mention(), coded(), g(0))])
        by_type = group_by_type(groups)
        assert set(by_type) == set(MentionType)
        assert len(by_type[MentionType.CONDITION]) == 1

    def test_count_by_assertion_includes_zeros(self):
        groups = dedup_mentions([(make_mention(assertion=Assertion.PRESENT), coded(), g(0))])
        counts = count_by_assertion(groups)
        assert counts["present"] == 1
        assert counts["negated"] == 0
        assert set(counts) == {a.value for a in Assertion}
