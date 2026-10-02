"""Section detection and chunking, including the offset round-trip invariant."""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from fhirground.exceptions import ConfigError
from fhirground.preprocess import (
    DEFAULT_SECTION_HEADERS,
    chunk_note,
    detect_sections,
    is_non_patient_section,
    section_at,
)

NOTE = """Chief Complaint: follow-up.

ASSESSMENT:
Type 2 diabetes, poorly controlled. A1c 9.2% today.
Start metformin 500 mg PO BID.

Family History:
Mother has hypertension.
"""


class TestSections:
    def test_detects_headers_in_order(self):
        names = [s.name for s in detect_sections(NOTE)]
        assert names == ["chief complaint", "assessment", "family history"]

    def test_case_and_spacing_tolerant(self):
        for variant in ("PAST MEDICAL HISTORY:", "Past Medical History :", "past medical history"):
            assert any(
                s.name == "past medical history" for s in detect_sections(f"{variant}\nDiabetes.\n")
            ), variant

    def test_longest_header_wins(self):
        sections = detect_sections("Assessment and Plan:\nStable.\n")
        assert sections[0].name == "assessment and plan"

    def test_section_spans_reach_the_next_header(self):
        sections = detect_sections(NOTE)
        assessment = next(s for s in sections if s.name == "assessment")
        family = next(s for s in sections if s.name == "family history")
        assert assessment.end == family.header_start

    def test_header_offset_belongs_to_its_section(self):
        """Chunks are cut at headers, so the header must belong somewhere."""
        sections = detect_sections(NOTE)
        for section in sections:
            assert section_at(sections, section.header_start) == section.name

    def test_body_lookup_excludes_the_header(self):
        sections = detect_sections(NOTE)
        assessment = next(s for s in sections if s.name == "assessment")
        assert not assessment.contains_body(assessment.header_start)
        assert assessment.contains_body(assessment.start + 1)

    def test_family_history_is_a_non_patient_section(self):
        assert is_non_patient_section("family history")
        assert not is_non_patient_section("assessment")
        assert not is_non_patient_section(None)

    def test_no_headers_is_not_an_error(self):
        assert detect_sections("Just a sentence with no headers.") == []

    def test_empty_text(self):
        assert detect_sections("") == []

    def test_custom_header_list(self):
        sections = detect_sections("Disposition:\nHome.\n", ("disposition",))
        assert sections[0].name == "disposition"

    def test_default_list_is_non_empty(self):
        assert len(DEFAULT_SECTION_HEADERS) > 20


class TestChunking:
    def test_short_note_is_one_chunk(self):
        plan = chunk_note("Short note.", "n1", max_chars=2000)
        assert len(plan) == 1
        assert plan.chunks[0].text == "Short note."

    def test_chunks_are_verbatim_slices(self):
        plan = chunk_note(NOTE, "n1", max_chars=80, overlap_chars=20)
        for chunk in plan:
            assert NOTE[chunk.start_offset : chunk.end_offset] == chunk.text

    def test_full_coverage(self):
        plan = chunk_note(NOTE, "n1", max_chars=80, overlap_chars=20)
        assert plan.coverage(len(NOTE)) == 1.0

    def test_chunks_are_labelled_with_their_section(self):
        plan = chunk_note(NOTE, "n1", max_chars=200, overlap_chars=20)
        assert {c.section for c in plan} == {
            "chief complaint",
            "assessment",
            "family history",
        }

    def test_chunks_do_not_span_sections_by_default(self):
        plan = chunk_note(NOTE, "n1", max_chars=5000)
        assert len(plan) == 3, "one chunk per section when sections are respected"

    def test_sections_can_be_ignored(self):
        plan = chunk_note(NOTE, "n1", max_chars=5000, respect_sections=False)
        assert len(plan) == 1

    def test_local_offsets_map_to_the_original(self):
        plan = chunk_note(NOTE, "n1", max_chars=80, overlap_chars=20)
        chunk = plan.chunks[1]
        for local in (0, 5, len(chunk.text)):
            assert chunk.to_original(local) == chunk.start_offset + local

    def test_out_of_range_local_offset_raises(self):
        plan = chunk_note("abc", "n1")
        with pytest.raises(ValueError, match="outside chunk"):
            plan.chunks[0].to_original(99)

    def test_overlap_repeats_text(self):
        text = " ".join(f"Sentence {i} here." for i in range(60))
        plan = chunk_note(text, "n1", max_chars=200, overlap_chars=60, respect_sections=False)
        assert len(plan) > 1
        for first, second in zip(plan.chunks, plan.chunks[1:], strict=False):
            assert second.start_offset < first.end_offset, "chunks must overlap"

    def test_chunk_ids_are_stable_and_ordered(self):
        plan = chunk_note(NOTE, "note-7", max_chars=80, overlap_chars=10)
        assert [c.chunk_id for c in plan] == [f"note-7#{i}" for i in range(len(plan))]

    def test_empty_text_yields_no_chunks(self):
        assert len(chunk_note("", "n1")) == 0

    @pytest.mark.parametrize(("max_chars", "overlap"), [(10, 0), (500, 500), (500, 600)])
    def test_impossible_parameters_rejected(self, max_chars, overlap):
        with pytest.raises(ConfigError):
            chunk_note(NOTE, "n1", max_chars=max_chars, overlap_chars=overlap)

    def test_negative_overlap_rejected(self):
        with pytest.raises(ConfigError, match="non-negative"):
            chunk_note(NOTE, "n1", overlap_chars=-5)

    def test_long_note_terminates_with_full_coverage(self):
        text = " ".join(f"Finding number {i} documented clearly." for i in range(500))
        plan = chunk_note(text, "n1", max_chars=400, overlap_chars=100, respect_sections=False)
        assert plan.coverage(len(text)) == 1.0
        assert all(text[c.start_offset : c.end_offset] == c.text for c in plan)

    def test_verify_against_catches_corruption(self):
        from dataclasses import replace

        plan = chunk_note("abcdefghij", "n1")
        bad = replace(plan.chunks[0], start_offset=3)
        with pytest.raises(ValueError, match="not a verbatim slice"):
            bad.verify_against("abcdefghij")


class TestChunkingProperties:
    """SPEC §8.1: offsets must round-trip for any text and parameters."""

    @settings(
        max_examples=60,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    @given(
        text=st.text(min_size=1, max_size=1200),
        max_chars=st.integers(min_value=50, max_value=400),
        overlap=st.integers(min_value=0, max_value=40),
    )
    def test_offsets_round_trip(self, text, max_chars, overlap):
        plan = chunk_note(text, "p", max_chars=max_chars, overlap_chars=overlap)
        for chunk in plan:
            assert text[chunk.start_offset : chunk.end_offset] == chunk.text
            for i in range(len(chunk.text)):
                assert text[chunk.start_offset + i] == chunk.text[i]

    @settings(
        max_examples=60,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    @given(
        text=st.text(min_size=1, max_size=1200),
        max_chars=st.integers(min_value=50, max_value=400),
        overlap=st.integers(min_value=0, max_value=40),
    )
    def test_coverage_is_complete(self, text, max_chars, overlap):
        plan = chunk_note(text, "p", max_chars=max_chars, overlap_chars=overlap)
        assert plan.coverage(len(text)) == pytest.approx(1.0)
