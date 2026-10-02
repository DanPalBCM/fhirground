"""The terminology interface: the only door a code may come through.

SPEC §0 states the hard rule -- the LLM never emits medical codes -- and this
module is how that rule is made structural rather than aspirational.

Two mechanisms, together:

1. The mention schema has no code fields, so a model has nowhere to put one.
2. Every mapping function accepts a :class:`Resolution` and **never a raw
   string**. A code therefore cannot reach a FHIR resource without having been
   constructed by a resolver, and a resolver only returns codes that came from
   its own value sets.

A property test drives adversarial mentions (text containing ``"SNOMED
44054006"``) through the pipeline and asserts that every ``coding.code`` in the
output exists in the active resolvers' value sets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from fhirground.schemas.mention import MentionType


@dataclass(frozen=True, slots=True)
class Coding:
    """A terminology code, with the system it belongs to.

    Frozen and constructed only by resolvers. ``display`` is the value set's
    own display string, never the model's wording -- otherwise a resource
    could carry a code from the terminology beside a label the model invented.
    """

    system: str
    code: str
    display: str | None = None

    def as_fhir(self) -> dict[str, Any]:
        """FHIR ``Coding`` element."""
        out: dict[str, Any] = {"system": self.system, "code": self.code}
        if self.display:
            out["display"] = self.display
        return out


@dataclass(frozen=True, slots=True)
class Resolution:
    """The outcome of resolving one mention's text to a code.

    A resolution with ``coding=None`` is a legitimate, expected result: SPEC
    design principle 3 says an unresolved mention becomes a text-only
    ``CodeableConcept`` and the confidence reflects it. Returning None-with-text
    rather than refusing to emit is what keeps recall usable without inventing
    codes.

    Attributes:
        text: The text that was resolved, preserved for ``code.text``.
        coding: The code, or None for a text-only concept.
        score: Match confidence in [0, 1]; the terminology component of the
            resource's confidence.
        resolver: Name of the resolver that produced this, for the debug trace.
        method: ``exact``, ``synonym``, ``fuzzy``, ``remote``, or ``none``.
        candidates: Runner-up matches, recorded for review when a fuzzy match
            was close.
    """

    text: str
    coding: Coding | None = None
    score: float = 0.0
    resolver: str = ""
    method: str = "none"
    candidates: tuple[tuple[str, float], ...] = field(default_factory=tuple)

    @property
    def is_coded(self) -> bool:
        """Whether a code was found."""
        return self.coding is not None

    def as_codeable_concept(self) -> dict[str, Any]:
        """FHIR ``CodeableConcept``, coded or text-only.

        ``text`` is always present. A consumer that cannot use the code can
        still read what the note said, and a text-only concept is visibly
        text-only rather than silently empty.
        """
        out: dict[str, Any] = {"text": self.text}
        if self.coding is not None:
            out["coding"] = [self.coding.as_fhir()]
        return out


@runtime_checkable
class TerminologyResolver(Protocol):
    """Anything that can turn mention text into a code.

    Implementations must return a :class:`Resolution` or None, and must never
    fabricate a code that is not in their own source data.
    """

    name: str

    def resolve(
        self, text: str, mention_type: MentionType, context: dict[str, Any] | None = None
    ) -> Resolution | None:
        """Resolve ``text`` to a code, or return None to pass to the next resolver."""
        ...

    def known_codes(self) -> set[tuple[str, str]]:
        """Every ``(system, code)`` this resolver could possibly emit.

        Used by the no-invented-codes property test: the union over active
        resolvers is the allowed set, and any code outside it is a bug. A
        network resolver returns what it has cached, and the test marks
        remote-resolved codes separately.
        """
        ...


@dataclass
class ResolverConfig:
    """Which resolver handles which mention type.

    Per-type rather than global because the terminologies differ: conditions go
    to SNOMED CT or ICD-10, medications to RxNorm, observations to LOINC. One
    resolver over all three would have to search every value set for every
    mention, which is both slower and more likely to cross-match.
    """

    condition: TerminologyResolver | None = None
    medication: TerminologyResolver | None = None
    observation: TerminologyResolver | None = None

    def for_type(self, mention_type: MentionType) -> TerminologyResolver | None:
        """The resolver for a mention type, or None."""
        return {
            MentionType.CONDITION: self.condition,
            MentionType.MEDICATION: self.medication,
            MentionType.OBSERVATION: self.observation,
        }[mention_type]

    def resolve(
        self,
        text: str,
        mention_type: MentionType,
        context: dict[str, Any] | None = None,
    ) -> Resolution:
        """Resolve through the configured resolver, falling back to text-only.

        Always returns a :class:`Resolution`, never None, so callers do not
        have to decide what an absent resolver means. With no resolver
        configured the result is text-only with score 0.0, which is the honest
        representation of "nobody looked".
        """
        resolver = self.for_type(mention_type)
        if resolver is None:
            return Resolution(text=text, resolver="none", method="none", score=0.0)
        found = resolver.resolve(text, mention_type, context)
        if found is None:
            return Resolution(text=text, resolver=resolver.name, method="none", score=0.0)
        return found

    def all_known_codes(self) -> set[tuple[str, str]]:
        """Union of every code the configured resolvers could emit."""
        out: set[tuple[str, str]] = set()
        for resolver in (self.condition, self.medication, self.observation):
            if resolver is not None:
                out |= resolver.known_codes()
        return out

    def describe(self) -> dict[str, str]:
        """Resolver name per mention type, for the run summary."""
        return {
            "condition": self.condition.name if self.condition else "none",
            "medication": self.medication.name if self.medication else "none",
            "observation": self.observation.name if self.observation else "none",
        }
