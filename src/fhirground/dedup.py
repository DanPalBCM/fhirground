"""Within-note deduplication.

Chunk overlap means the same sentence can be extracted twice, and a note may
state the same finding in two places. Both produce duplicate mentions, and
emitting two Conditions for one diagnosis would overstate what the note says.

Mentions merge when they agree on all three of:

1. **type** -- a condition and an observation are never the same thing;
2. **normalized concept** -- the resolved code when there is one, else the
   normalized text. Using the code when available is what lets "T2DM" and
   "type 2 diabetes" merge;
3. **assertion** -- "possible pneumonia" and "pneumonia" are different claims
   and must not collapse into one.

**Every evidence span survives the merge.** A merged resource carries one
evidence extension per source mention, so the audit trail does not shrink when
the resource count does -- which is the difference between deduplication and
data loss.

Cross-note reconciliation is explicitly out of scope: two notes mentioning the
same diagnosis are two independent pieces of evidence, and merging them is a
patient-level problem with its own failure modes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fhirground.grounding import Grounding
from fhirground.schemas.mention import Assertion, Mention, MentionType
from fhirground.terminology.base import Resolution


@dataclass
class MentionGroup:
    """One deduplicated finding and every mention that supported it.

    Attributes:
        key: The identity the group merged on.
        mention: The representative mention -- the one with the best grounding,
            so the emitted resource quotes the clearest evidence.
        resolution: The representative resolution.
        groundings: Every evidence span, in document order.
        members: Every mention in the group.
        n_duplicates: How many extra mentions merged in.
    """

    key: tuple[str, str, str]
    mention: Mention
    resolution: Resolution
    groundings: list[Grounding] = field(default_factory=list)
    members: list[Mention] = field(default_factory=list)

    @property
    def n_duplicates(self) -> int:
        """Mentions beyond the first."""
        return max(0, len(self.members) - 1)

    @property
    def best_grounding(self) -> Grounding:
        """The highest-scoring span, used for the representative evidence."""
        return max(self.groundings, key=lambda g: (g.score, -g.start))

    def as_debug(self) -> dict[str, Any]:
        """Summary for the debug trace."""
        return {
            "key": list(self.key),
            "n_members": len(self.members),
            "n_duplicates": self.n_duplicates,
            "spans": [[g.start, g.end] for g in self.groundings],
            "strategies": [g.strategy.value for g in self.groundings],
        }


def dedup_key(mention: Mention, resolution: Resolution) -> tuple[str, str, str]:
    """The identity a mention merges on.

    The concept part prefers the resolved code, because that is what makes
    different surface forms of one concept merge. Falling back to normalized
    text means unresolved mentions still deduplicate against *themselves*,
    just not against synonyms -- which is the correct, conservative outcome
    when no terminology was available to tell you they were the same.
    """
    if resolution.coding is not None:
        concept = f"{resolution.coding.system}|{resolution.coding.code}"
    else:
        concept = f"text|{mention.normalized_text}"
    return (mention.mention_type.value, concept, mention.assertion.value)


def dedup_mentions(
    items: list[tuple[Mention, Resolution, Grounding]],
) -> list[MentionGroup]:
    """Group mentions that denote the same finding.

    Args:
        items: ``(mention, resolution, grounding)`` triples, in extraction
            order.

    Returns:
        Groups in first-appearance order, each with every evidence span kept.
        Order is stable so that resource ids are deterministic across runs.
    """
    groups: dict[tuple[str, str, str], MentionGroup] = {}

    for mention, resolution, grounding in items:
        key = dedup_key(mention, resolution)
        existing = groups.get(key)
        if existing is None:
            groups[key] = MentionGroup(
                key=key,
                mention=mention,
                resolution=resolution,
                groundings=[grounding],
                members=[mention],
            )
            continue

        existing.members.append(mention)

        # Capture the incumbent's best score *before* adding the new span.
        # Appending first would put the new grounding inside the max, so it
        # could never beat it and the representative would never be promoted.
        incumbent_score = existing.best_grounding.score

        # Keep spans unique: overlap can yield the identical span twice, and
        # two identical evidence extensions add nothing.
        if not any(
            g.start == grounding.start and g.end == grounding.end for g in existing.groundings
        ):
            existing.groundings.append(grounding)

        # Promote the representative when the new mention is better grounded,
        # or when it carries detail the incumbent lacked. A mention with a dose
        # is more useful than the same mention without one.
        if grounding.score > incumbent_score or _detail_score(mention) > _detail_score(
            existing.mention
        ):
            existing.mention = mention
            if resolution.is_coded and not existing.resolution.is_coded:
                existing.resolution = resolution
        elif resolution.is_coded and not existing.resolution.is_coded:
            existing.resolution = resolution

    for group in groups.values():
        group.groundings.sort(key=lambda g: g.start)
    return list(groups.values())


def _detail_score(mention: Mention) -> int:
    """How much structured detail a mention carries.

    Used only as a tie-break for which mention represents a group. More
    populated optional fields means a richer resource.
    """
    fields = (
        mention.clinical_status_hint,
        mention.onset_text,
        mention.med_action,
        mention.dose_value,
        mention.dose_unit,
        mention.route,
        mention.frequency_text,
        mention.obs_category,
        mention.value_number,
        mention.value_comparator,
        mention.value_text,
        mention.value_unit,
        mention.date_text,
    )
    return sum(1 for f in fields if f is not None)


def matches_for_consistency(
    target: tuple[Mention, Resolution], samples: list[list[tuple[Mention, Resolution]]]
) -> int:
    """How many samples contain a mention matching ``target``.

    This is the self-consistency component of confidence (SPEC §4.6): with
    ``k`` samples at a non-zero temperature, a finding every sample agrees on
    is more trustworthy than one that appeared once.

    Matching uses the same three-part identity as deduplication, so
    "consistent" means the same thing in both places.
    """
    mention, resolution = target
    key = dedup_key(mention, resolution)
    count = 0
    for sample in samples:
        if any(dedup_key(m, r) == key for m, r in sample):
            count += 1
    return count


def group_by_type(groups: list[MentionGroup]) -> dict[MentionType, list[MentionGroup]]:
    """Partition groups by mention type, for per-type reporting."""
    out: dict[MentionType, list[MentionGroup]] = {t: [] for t in MentionType}
    for group in groups:
        out[MentionType(group.key[0])].append(group)
    return out


def count_by_assertion(groups: list[MentionGroup]) -> dict[str, int]:
    """Group counts per assertion value, for the run summary."""
    counts = {a.value: 0 for a in Assertion}
    for group in groups:
        counts[group.key[2]] = counts.get(group.key[2], 0) + 1
    return counts
