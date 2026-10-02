"""Versioned prompt templates.

The ``prompt_version`` string goes into every ``Provenance``, so a result can
be traced to the exact instructions that produced it. Changing a template
without bumping the version makes two runs indistinguishable in the output
while actually differing -- so the version is part of the template constant,
not a parameter.

The prompt tells the model what *not* to do as explicitly as what to do,
because the failure modes are predictable: inventing codes, paraphrasing the
evidence quote, and reporting family history as the patient's own.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Current prompt version. Bump on any change to the template text.
PROMPT_VERSION = "mention-v1"

_SYSTEM = """\
You extract clinical findings from notes. You return JSON and nothing else.

Rules:

1. Extract three kinds of finding: `condition`, `medication`, `observation`.
2. `evidence` must be a VERBATIM substring of the note. Copy it exactly,
   including punctuation and capitalization. Do not paraphrase, summarize, or
   correct it. If you cannot copy it exactly, do not report the finding.
3. `text` is the concept in plain language as the note expresses it.
4. Never output a medical code. There is no field for one. Do not put SNOMED,
   ICD-10, LOINC, or RxNorm codes in `text` or any other field, even if the
   note contains one.
5. Set `assertion` honestly:
   - `present`: the note asserts the patient has it now
   - `negated`: the note denies it ("denies chest pain", "no fever")
   - `possible`: hedged ("possible pneumonia", "rule out sepsis")
   - `historical`: the patient had it in the past
   - `family`: someone else in the family has it, not the patient
   - `hypothetical`: conditional or planned ("if she develops a rash")
6. Leave an optional field `null` rather than guessing. A missing dose is
   better than an invented one.
7. Report each distinct finding once. Do not repeat a finding to fill space.
"""

_USER = """\
{context}Extract every clinical finding from the note text below.

Return JSON matching the schema exactly. If the note contains no findings,
return {{"mentions": []}}.

--- NOTE TEXT START ---
{chunk}
--- NOTE TEXT END ---
"""

_REPAIR = """\
Your previous response was rejected.

Error:
{error}

Return corrected JSON matching the schema exactly. Change only what the error
requires. Do not add findings that were not in the note, and keep every
`evidence` value a verbatim substring of the note text.
"""


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    """A prompt ready for a backend."""

    system: str
    user: str
    version: str = PROMPT_VERSION

    @property
    def combined(self) -> str:
        """System and user joined, for backends with no system-message slot."""
        return f"{self.system}\n\n{self.user}"

    def digest(self) -> str:
        """Short hash of the full prompt, for the LLM call cache key."""
        import hashlib

        return hashlib.sha256(self.combined.encode("utf-8")).hexdigest()[:16]


def render(
    chunk_text: str,
    *,
    section: str | None = None,
    note_datetime: str | None = None,
) -> RenderedPrompt:
    """Render the extraction prompt for one chunk.

    Args:
        chunk_text: The verbatim chunk.
        section: Section the chunk came from, supplied as context so the model
            can read a bare "hypertension" under *Family History* correctly.
        note_datetime: The note's timestamp, so relative phrases can be
            reported as written and resolved later by code.

    Returns:
        A :class:`RenderedPrompt`.
    """
    lines: list[str] = []
    if section:
        lines.append(f"This text comes from the note's {section!r} section.")
    if note_datetime:
        lines.append(f"The note was written on {note_datetime}.")
    context = ("\n".join(lines) + "\n\n") if lines else ""
    return RenderedPrompt(system=_SYSTEM, user=_USER.format(context=context, chunk=chunk_text))


def render_repair(previous: RenderedPrompt, error: str) -> RenderedPrompt:
    """Render the repair prompt after a parse or schema failure.

    The error text is fed back verbatim because it names the offending field
    and what was expected -- which is exactly what a model needs to fix it.
    """
    return RenderedPrompt(
        system=previous.system,
        user=f"{previous.user}\n\n{_REPAIR.format(error=error)}",
        version=previous.version,
    )
