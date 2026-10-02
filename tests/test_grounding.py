"""Grounding: the three strategies, rejection, ambiguity, and multi-chunk spans."""

from __future__ import annotations

from fhirground.grounding import (
    DEFAULT_FUZZY_THRESHOLD,
    GroundingStrategy,
    ground_in_chunk,
    ground_mention,
    normalize,
)
from fhirground.preprocess import chunk_note

NOTE = (
    "ASSESSMENT:\nType 2 diabetes, poorly controlled. A1c 9.2% today. "
    "Start metformin 500 mg PO BID.\nDenies chest pain."
)


def chunks(text: str = NOTE, **kwargs):
    return chunk_note(text, "n1", **{"max_chars": 4000, **kwargs}).chunks


class TestExact:
    def test_exact_hit_scores_one(self):
        g = ground_mention("Type 2 diabetes, poorly controlled", chunks(), NOTE)
        assert g.strategy is GroundingStrategy.EXACT
        assert g.score == 1.0
        assert g.verify(NOTE)

    def test_offsets_are_original_text_offsets(self):
        g = ground_mention("Denies chest pain", chunks(), NOTE)
        assert NOTE[g.start : g.end] == "Denies chest pain"


class TestNormalized:
    def test_case_and_whitespace_differences_still_ground(self):
        g = ground_mention("type 2  DIABETES,   poorly controlled", chunks(), NOTE)
        assert g.strategy is GroundingStrategy.NORMALIZED
        assert g.score == 0.9

    def test_stored_quote_is_what_the_note_says(self):
        """Not the model's wording -- otherwise the offset/quote rule is unverifiable."""
        g = ground_mention("type 2  DIABETES,   poorly controlled", chunks(), NOTE)
        assert g.quote == "Type 2 diabetes, poorly controlled"
        assert g.verify(NOTE)

    def test_normalize_helper(self):
        assert normalize("  A   B\nC  ") == "a b c"


class TestFuzzy:
    def test_near_miss_grounds_to_the_right_place(self):
        g = ground_mention("Start metformin 500mg PO BID", chunks(), NOTE)
        assert g.strategy is GroundingStrategy.FUZZY
        assert "metformin" in g.quote
        assert g.verify(NOTE)

    def test_score_scales_with_the_ratio(self):
        g = ground_mention("Start metformin 500mg PO BID", chunks(), NOTE)
        assert 0.0 < g.score <= 0.8

    def test_below_threshold_is_rejected(self):
        assert ground_mention("fractured left femur requiring surgery", chunks(), NOTE) is None

    def test_threshold_is_configurable(self):
        loose = ground_mention("chest pian", chunks(), NOTE, fuzzy_threshold=50.0)
        assert loose is not None
        strict = ground_mention("chest pian", chunks(), NOTE, fuzzy_threshold=99.0)
        assert strict is None

    def test_default_threshold_is_the_spec_value(self):
        assert DEFAULT_FUZZY_THRESHOLD == 90.0


class TestRejection:
    def test_empty_evidence(self):
        assert ground_mention("", chunks(), NOTE) is None
        assert ground_mention("   ", chunks(), NOTE) is None

    def test_absent_text(self):
        assert ground_mention("pulmonary embolism on CT angiogram", chunks(), NOTE) is None


class TestAmbiguity:
    def test_repeated_evidence_takes_the_first_and_records_it(self):
        text = "Chest pain noted. Later: Chest pain noted."
        g = ground_mention("Chest pain noted.", chunks(text), text)
        assert g.start == 0, "the first occurrence wins"
        assert g.ambiguous is True

    def test_unique_evidence_is_not_ambiguous(self):
        g = ground_mention("Denies chest pain", chunks(), NOTE)
        assert g.ambiguous is False


class TestMultiChunk:
    def test_preferred_chunk_is_tried_first(self):
        text = " ".join(f"Finding {i} recorded." for i in range(40))
        plan = chunk_note(text, "n1", max_chars=200, overlap_chars=80, respect_sections=False)
        assert len(plan) > 1
        target = "Finding 20 recorded."
        g = ground_mention(target, plan.chunks, text, preferred_chunk_id=plan.chunks[-1].chunk_id)
        assert g is not None
        assert text[g.start : g.end] == target

    def test_evidence_found_in_a_later_chunk(self):
        text = " ".join(f"Finding {i} recorded." for i in range(40))
        plan = chunk_note(text, "n1", max_chars=150, overlap_chars=40, respect_sections=False)
        g = ground_mention(
            "Finding 38 recorded.", plan.chunks, text, preferred_chunk_id=plan.chunks[0].chunk_id
        )
        assert g is not None
        assert text[g.start : g.end] == "Finding 38 recorded."

    def test_chunk_id_is_recorded(self):
        plan = chunk_note(NOTE, "n1", max_chars=4000)
        g = ground_in_chunk("Denies chest pain", plan.chunks[0], NOTE)
        assert g.chunk_id == plan.chunks[0].chunk_id


class TestGroundingShape:
    def test_length_and_verify(self):
        g = ground_mention("Denies chest pain", chunks(), NOTE)
        assert g.length == len("Denies chest pain")
        assert g.verify(NOTE)
        assert not g.verify("completely different text")

    def test_strategy_scores_are_ordered(self):
        exact = ground_mention("Denies chest pain", chunks(), NOTE)
        norm = ground_mention("denies  CHEST pain", chunks(), NOTE)
        fuzzy = ground_mention("Start metformin 500mg PO BID", chunks(), NOTE)
        assert exact.score > norm.score > fuzzy.score
