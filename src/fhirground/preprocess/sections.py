"""Section-header detection.

Regex-based and configurable. Not a clinical document parser: it finds lines
that look like ``ASSESSMENT:`` or ``Past Medical History:`` and records where
they start. The section a mention came from is useful context for assertion
(a finding under "Family History" is family history even when the sentence does
not say so) and it goes into the debug trace.

Deliberately conservative. A missed header costs a little context; a false
header could split a sentence across chunks and break grounding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Headers fhirground recognizes by default. Matching is case-insensitive and
#: tolerant of spacing, so "PAST MEDICAL HISTORY:" and "Past Medical History :"
#: both match. Extend via PipelineConfig.section_headers.
DEFAULT_SECTION_HEADERS: tuple[str, ...] = (
    "chief complaint",
    "history of present illness",
    "hpi",
    "past medical history",
    "pmh",
    "past surgical history",
    "family history",
    "social history",
    "review of systems",
    "ros",
    "medications",
    "current medications",
    "medications on admission",
    "discharge medications",
    "allergies",
    "physical exam",
    "physical examination",
    "vital signs",
    "vitals",
    "laboratory",
    "labs",
    "lab results",
    "results",
    "imaging",
    "assessment",
    "assessment and plan",
    "impression",
    "plan",
    "diagnosis",
    "diagnoses",
    "hospital course",
    "discharge diagnosis",
    "subjective",
    "objective",
)

#: Sections whose contents should never be read as the patient's own active
#: findings. Used by the assertion policy as a contextual signal.
NON_PATIENT_SECTIONS: frozenset[str] = frozenset({"family history"})


@dataclass(frozen=True, slots=True)
class Section:
    """A detected section header and the span of text under it.

    Attributes:
        name: Normalized header name, lower-cased.
        header_start: Offset of the header in the original text.
        start: Offset where the section body begins (just after the header).
        end: Offset where the body ends, which is the next header or the end
            of the note.
    """

    name: str
    header_start: int
    start: int
    end: int

    def contains(self, offset: int) -> bool:
        """Whether an original-text offset falls under this section.

        Measured from ``header_start``, not ``start``, so the header line
        itself belongs to the section it names. Chunks are cut *at* header
        boundaries, so a body-only range would leave every chunk's first
        offset belonging to no section at all.
        """
        return self.header_start <= offset < self.end

    def contains_body(self, offset: int) -> bool:
        """Whether an offset falls in the section's body, excluding its header."""
        return self.start <= offset < self.end


def _build_pattern(headers: tuple[str, ...]) -> re.Pattern[str]:
    """Compile an alternation that matches any configured header.

    Longest-first so that ``assessment and plan`` wins over ``assessment``;
    otherwise the shorter alternative matches and the section name is wrong.
    """
    ordered = sorted({h.strip().lower() for h in headers if h.strip()}, key=len, reverse=True)
    alternation = "|".join(re.escape(h).replace(r"\ ", r"\s+") for h in ordered)
    # A header must start a line and be followed by a colon or a line break.
    return re.compile(
        rf"^[ \t]*(?P<name>{alternation})[ \t]*:?[ \t]*$|^[ \t]*(?P<name2>{alternation})[ \t]*:",
        re.IGNORECASE | re.MULTILINE,
    )


def detect_sections(text: str, headers: tuple[str, ...] = DEFAULT_SECTION_HEADERS) -> list[Section]:
    """Find section headers and the spans they govern.

    Args:
        text: The original note text.
        headers: Header names to look for.

    Returns:
        Sections in document order. Empty when no header matched -- a note
        with no headers is normal and is treated as one unnamed span by
        callers.
    """
    if not text or not headers:
        return []

    pattern = _build_pattern(tuple(headers))
    found: list[tuple[str, int, int]] = []
    for match in pattern.finditer(text):
        name = (match.group("name") or match.group("name2") or "").strip().lower()
        name = " ".join(name.split())
        found.append((name, match.start(), match.end()))

    sections: list[Section] = []
    for i, (name, header_start, body_start) in enumerate(found):
        end = found[i + 1][1] if i + 1 < len(found) else len(text)
        sections.append(Section(name=name, header_start=header_start, start=body_start, end=end))
    return sections


def section_at(sections: list[Section], offset: int) -> str | None:
    """Name of the section containing ``offset``, or None.

    Searched in reverse so the innermost (latest-starting) section wins when
    spans touch.
    """
    for section in reversed(sections):
        if section.contains(offset):
            return section.name
    return None


def is_non_patient_section(name: str | None) -> bool:
    """Whether a section name implies the findings are not the patient's own."""
    return name is not None and name in NON_PATIENT_SECTIONS
