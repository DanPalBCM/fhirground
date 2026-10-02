"""Extension URLs and builders.

Three extensions carry everything fhirground adds beyond plain FHIR:

* ``extraction-confidence`` -- the score, and whether it was calibrated.
* ``evidence`` -- where in the source note the claim came from.
* ``effective-inferred`` -- that an Observation's time came from the note
  timestamp rather than from the note text.

The base URL is **configurable and has no default that pretends to be real**.
Publishing resources with someone else's StructureDefinition URL would be
worse than publishing a placeholder, so the placeholder is explicit and the
docs tell you to set it.

The StructureDefinitions themselves are not shipped in v1; the URLs identify
the extensions and the docs define their semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fhirground.grounding import Grounding

#: Placeholder base. Replace via ``PipelineConfig.extension_base_url``.
DEFAULT_EXTENSION_BASE = "https://github.com/DanPalBCM/fhirground/StructureDefinition/"

#: Extension name suffixes, appended to the base URL.
CONFIDENCE_EXT = "extraction-confidence"
EVIDENCE_EXT = "evidence"
EFFECTIVE_INFERRED_EXT = "effective-inferred"

#: Sub-extension names inside the complex extensions.
CALIBRATED_SUB = "calibrated"
EVIDENCE_SOURCE_SUB = "source"
EVIDENCE_START_SUB = "start"
EVIDENCE_END_SUB = "end"
EVIDENCE_QUOTE_SUB = "quote"
EVIDENCE_GROUNDING_SUB = "grounding"


@dataclass(frozen=True, slots=True)
class ExtensionUrls:
    """Resolved extension URLs for one configured base."""

    base: str = DEFAULT_EXTENSION_BASE

    def __post_init__(self) -> None:
        if not self.base.endswith("/"):
            object.__setattr__(self, "base", self.base + "/")

    @property
    def confidence(self) -> str:
        """URL of the ``extraction-confidence`` extension."""
        return f"{self.base}{CONFIDENCE_EXT}"

    @property
    def evidence(self) -> str:
        """URL of the ``evidence`` extension."""
        return f"{self.base}{EVIDENCE_EXT}"

    @property
    def effective_inferred(self) -> str:
        """URL of the ``effective-inferred`` extension."""
        return f"{self.base}{EFFECTIVE_INFERRED_EXT}"

    def all(self) -> tuple[str, ...]:
        """Every URL this configuration produces."""
        return (self.confidence, self.evidence, self.effective_inferred)


def confidence_extension(
    score: float, *, calibrated: bool = False, urls: ExtensionUrls | None = None
) -> dict[str, Any]:
    """Build the ``extraction-confidence`` extension.

    Args:
        score: The confidence in [0, 1].
        calibrated: Whether a fitted calibrator produced it. **This flag is
            the difference between a probability and a heuristic**, so it is
            recorded on every resource rather than only in the run summary.
        urls: Configured extension URLs.

    Raises:
        ValueError: If the score is outside [0, 1], which would make the
            extension meaningless.
    """
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"confidence must be in [0, 1], got {score}")
    u = urls or ExtensionUrls()
    return {
        "url": u.confidence,
        "valueDecimal": round(float(score), 4),
        "extension": [{"url": CALIBRATED_SUB, "valueBoolean": bool(calibrated)}],
    }


def evidence_extension(
    grounding: Grounding,
    *,
    source_reference: str | None = None,
    include_quote: bool = True,
    urls: ExtensionUrls | None = None,
) -> dict[str, Any]:
    """Build one ``evidence`` extension from a grounding result.

    Args:
        grounding: Where the claim came from.
        source_reference: Reference to the source ``DocumentReference``.
        include_quote: When False the quote is omitted for PHI minimization,
            but the offsets stay. SPEC §4.5 still requires offsets to be in
            range, so the evidence remains checkable against the note even
            when the text is not carried in the resource.
        urls: Configured extension URLs.
    """
    u = urls or ExtensionUrls()
    parts: list[dict[str, Any]] = []
    if source_reference:
        parts.append(
            {"url": EVIDENCE_SOURCE_SUB, "valueReference": {"reference": source_reference}}
        )
    parts.append({"url": EVIDENCE_START_SUB, "valueInteger": int(grounding.start)})
    parts.append({"url": EVIDENCE_END_SUB, "valueInteger": int(grounding.end)})
    if include_quote:
        parts.append({"url": EVIDENCE_QUOTE_SUB, "valueString": grounding.quote})
    parts.append({"url": EVIDENCE_GROUNDING_SUB, "valueCode": grounding.strategy.value})
    return {"url": u.evidence, "extension": parts}


def effective_inferred_extension(
    inferred: bool = True, *, urls: ExtensionUrls | None = None
) -> dict[str, Any]:
    """Build the ``effective-inferred`` extension for an Observation."""
    u = urls or ExtensionUrls()
    return {"url": u.effective_inferred, "valueBoolean": bool(inferred)}


def read_confidence(resource: dict[str, Any], urls: ExtensionUrls | None = None) -> float | None:
    """Read the confidence back off a resource, or None if absent."""
    u = urls or ExtensionUrls()
    for ext in resource.get("extension") or []:
        if ext.get("url") == u.confidence:
            value = ext.get("valueDecimal")
            return float(value) if value is not None else None
    return None


def read_evidence(
    resource: dict[str, Any], urls: ExtensionUrls | None = None
) -> list[dict[str, Any]]:
    """Read every evidence extension off a resource as flat dicts.

    Returns:
        One dict per evidence extension with ``start``, ``end``, and whichever
        of ``quote``, ``grounding``, and ``source`` were present.
    """
    u = urls or ExtensionUrls()
    out: list[dict[str, Any]] = []
    for ext in resource.get("extension") or []:
        if ext.get("url") != u.evidence:
            continue
        flat: dict[str, Any] = {}
        for sub in ext.get("extension") or []:
            key = sub.get("url")
            if key == EVIDENCE_START_SUB:
                flat["start"] = sub.get("valueInteger")
            elif key == EVIDENCE_END_SUB:
                flat["end"] = sub.get("valueInteger")
            elif key == EVIDENCE_QUOTE_SUB:
                flat["quote"] = sub.get("valueString")
            elif key == EVIDENCE_GROUNDING_SUB:
                flat["grounding"] = sub.get("valueCode")
            elif key == EVIDENCE_SOURCE_SUB:
                flat["source"] = (sub.get("valueReference") or {}).get("reference")
        out.append(flat)
    return out


def read_calibrated(resource: dict[str, Any], urls: ExtensionUrls | None = None) -> bool | None:
    """Whether the resource's confidence was calibrated, or None if absent."""
    u = urls or ExtensionUrls()
    for ext in resource.get("extension") or []:
        if ext.get("url") == u.confidence:
            for sub in ext.get("extension") or []:
                if sub.get("url") == CALIBRATED_SUB:
                    return bool(sub.get("valueBoolean"))
    return None
