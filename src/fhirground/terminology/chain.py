"""Chaining resolvers: try each in turn, keep the first hit.

The practical arrangement is a local value set first and a network resolver
second. Local is fast, deterministic, offline, and governed by whatever
terminology the site is licensed for; the network is a fallback for the long
tail. Putting them the other way round would make every run depend on a remote
service and make results irreproducible.

A chain can also be configured to keep the **best** hit rather than the first,
for comparing resolvers.
"""

from __future__ import annotations

from typing import Any

from fhirground.exceptions import TerminologyError
from fhirground.schemas.mention import MentionType
from fhirground.terminology.base import Resolution, TerminologyResolver


class Chain:
    """Several resolvers tried in order.

    Args:
        resolvers: Resolvers in priority order.
        name: Name recorded on resolutions and in the run summary.
        strategy: ``first`` keeps the first hit (the default, and fastest);
            ``best`` queries every resolver and keeps the highest score.
        min_score: A hit below this score is treated as a miss and the chain
            continues. Useful when a local value set returns weak fuzzy
            matches that a remote lookup would beat.
    """

    def __init__(
        self,
        resolvers: list[TerminologyResolver],
        *,
        name: str | None = None,
        strategy: str = "first",
        min_score: float = 0.0,
    ) -> None:
        if not resolvers:
            raise TerminologyError(
                "a Chain needs at least one resolver; use NullResolver for text-only output"
            )
        if strategy not in ("first", "best"):
            raise TerminologyError(f"unknown chain strategy {strategy!r}; use 'first' or 'best'")
        self.resolvers = resolvers
        self.strategy = strategy
        self.min_score = min_score
        self.name = name or "chain(" + "+".join(r.name for r in resolvers) + ")"

    def __len__(self) -> int:
        return len(self.resolvers)

    def resolve(
        self,
        text: str,
        mention_type: MentionType,
        context: dict[str, Any] | None = None,
    ) -> Resolution | None:
        """Resolve through the chain.

        Returns:
            The chosen :class:`Resolution` with its originating resolver
            recorded, or None when every resolver missed.
        """
        best: Resolution | None = None
        for resolver in self.resolvers:
            found = resolver.resolve(text, mention_type, context)
            if found is None or not found.is_coded:
                continue
            if found.score < self.min_score:
                continue
            if self.strategy == "first":
                return found
            if best is None or found.score > best.score:
                best = found
        return best

    def known_codes(self) -> set[tuple[str, str]]:
        """Union of every code any resolver in the chain could emit."""
        out: set[tuple[str, str]] = set()
        for resolver in self.resolvers:
            out |= resolver.known_codes()
        return out
