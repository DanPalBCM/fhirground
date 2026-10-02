"""The pipeline: note in, validated FHIR Bundle out.

Stage order is SPEC §4.2, and it is the order that makes the guarantees hold:

    chunk → extract → ground → assert → normalize → resolve → dedup
          → map → validate → score → bundle

Grounding comes **before** the assertion policy so an ungrounded mention is
dropped before anything decides what it means. Terminology resolution comes
**before** deduplication so that two surface forms of one concept merge on
their shared code. Validation comes **before** confidence so a resource that
fails a project rule is dropped rather than scored.

Every dropped mention is counted with a reason, and every emitted resource
carries its evidence. A note that produces nothing produces a reason for it.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from fhirground.assertion import DropReason, EmissionDecision, decide
from fhirground.backends.base import Backend
from fhirground.backends.prompts import PROMPT_VERSION
from fhirground.confidence import (
    Calibrator,
    ConfidenceComponents,
    aggregate,
    self_consistency,
    validation_component,
)
from fhirground.dedup import MentionGroup, dedup_mentions, matches_for_consistency
from fhirground.exceptions import ConfigError
from fhirground.extract import CallCache, extract_chunk
from fhirground.grounding import DEFAULT_FUZZY_THRESHOLD, Grounding, ground_mention
from fhirground.io import NoteInput
from fhirground.mapping import (
    ExtensionUrls,
    build_bundle,
    build_condition,
    build_medication_request,
    build_observation,
    build_provenance,
)
from fhirground.mapping.condition import add_evidence as add_condition_evidence
from fhirground.mapping.medication_request import (
    add_evidence as add_med_evidence,
)
from fhirground.mapping.observation import add_evidence as add_obs_evidence
from fhirground.normalize.dates import inferred_from_note, resolve_date
from fhirground.normalize.values import parse_value
from fhirground.preprocess.chunking import chunk_note
from fhirground.preprocess.sections import DEFAULT_SECTION_HEADERS, section_at
from fhirground.schemas.jsonschema import build_mention_schema, schema_digest
from fhirground.schemas.mention import Mention, MentionType
from fhirground.terminology.base import Resolution, ResolverConfig
from fhirground.validate import ValidationReport, validate_resource


class PipelineConfig(BaseModel):
    """Everything that changes what the pipeline produces."""

    model_config = ConfigDict(extra="forbid")

    # extraction
    samples: int = Field(default=1, ge=1, le=16)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_repairs: int = Field(default=2, ge=0, le=5)

    # chunking
    max_chars: int = Field(default=2000, ge=50)
    overlap_chars: int = Field(default=200, ge=0)
    section_headers: tuple[str, ...] = DEFAULT_SECTION_HEADERS
    respect_sections: bool = True

    # grounding
    fuzzy_threshold: float = Field(default=DEFAULT_FUZZY_THRESHOLD, ge=0.0, le=100.0)

    # emission policy
    emit_refuted: bool = False
    respect_non_patient_sections: bool = True
    include_quotes: bool = True
    include_category: bool = False
    mark_inferred_effective: bool = True
    infer_observation_effective: bool = True

    # scoring
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    weights: dict[str, float] | None = None

    # output
    extension_base_url: str | None = None

    # caching
    cache_path: Path | None = None

    def validate_consistency(self) -> None:
        """Catch combinations that cannot do what the caller intends.

        Raises:
            ConfigError: When the configuration is self-defeating rather than
                merely unusual.
        """
        if self.overlap_chars >= self.max_chars:
            raise ConfigError(
                f"overlap_chars ({self.overlap_chars}) must be smaller than max_chars "
                f"({self.max_chars}), or chunking cannot advance"
            )
        if self.samples > 1 and self.temperature == 0.0:
            raise ConfigError(
                f"samples={self.samples} with temperature=0.0 produces identical "
                "samples, so the self-consistency score would always be 1.0 and mean "
                "nothing. Raise the temperature or set samples=1."
            )

    @property
    def extension_urls(self) -> ExtensionUrls:
        """Resolved extension URLs."""
        return (
            ExtensionUrls(self.extension_base_url) if self.extension_base_url else ExtensionUrls()
        )


@dataclass
class MentionTrace:
    """Everything that happened to one mention, for the debug output."""

    mention: Mention
    chunk_id: str
    grounding: Grounding | None = None
    resolution: Resolution | None = None
    decision: EmissionDecision | None = None
    components: ConfidenceComponents | None = None
    score: float | None = None
    calibrated: bool = False
    resource_id: str | None = None
    resource_type: str | None = None
    drop_reason: DropReason | None = None
    validation: ValidationReport | None = None
    n_duplicates: int = 0

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe trace row."""
        return {
            "mention": self.mention.model_dump(mode="json", exclude_none=True),
            "chunk_id": self.chunk_id,
            "grounding": (
                {
                    "start": self.grounding.start,
                    "end": self.grounding.end,
                    "strategy": self.grounding.strategy.value,
                    "score": self.grounding.score,
                    "ambiguous": self.grounding.ambiguous,
                }
                if self.grounding
                else None
            ),
            "resolution": (
                {
                    "coded": self.resolution.is_coded,
                    "system": self.resolution.coding.system if self.resolution.coding else None,
                    "code": self.resolution.coding.code if self.resolution.coding else None,
                    "score": self.resolution.score,
                    "resolver": self.resolution.resolver,
                    "method": self.resolution.method,
                }
                if self.resolution
                else None
            ),
            "decision": (
                {
                    "emit": self.decision.emit,
                    "reason": self.decision.reason.value if self.decision.reason else None,
                    "note": self.decision.note,
                }
                if self.decision
                else None
            ),
            "confidence_components": self.components.as_dict() if self.components else None,
            "score": self.score,
            "calibrated": self.calibrated,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "drop_reason": self.drop_reason.value if self.drop_reason else None,
            "n_duplicates": self.n_duplicates,
            "validation": self.validation.as_dict() if self.validation else None,
        }


@dataclass
class NoteResult:
    """What one note produced."""

    note: NoteInput
    bundle: dict[str, Any] = field(default_factory=dict)
    resources: list[dict[str, Any]] = field(default_factory=list)
    dropped: list[tuple[Mention, DropReason]] = field(default_factory=list)
    debug: list[MentionTrace] = field(default_factory=list)
    validation: ValidationReport = field(default_factory=ValidationReport)
    stats: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether the note processed without a hard failure."""
        return not self.errors

    def by_type(self, resource_type: str) -> list[dict[str, Any]]:
        """Emitted resources of one type."""
        return [r for r in self.resources if r.get("resourceType") == resource_type]

    def drop_counts(self) -> dict[str, int]:
        """Dropped mention counts per reason, every reason present."""
        counts = {r.value: 0 for r in DropReason}
        for _mention, reason in self.dropped:
            counts[reason.value] = counts.get(reason.value, 0) + 1
        return counts

    def resource_counts(self) -> dict[str, int]:
        """Emitted resource counts per type."""
        counts: dict[str, int] = {}
        for resource in self.resources:
            key = str(resource.get("resourceType"))
            counts[key] = counts.get(key, 0) + 1
        return counts


class Pipeline:
    """Wires preprocessing, extraction, mapping, validation, and scoring.

    Example::

        pipe = Pipeline(PipelineConfig(), backend=MockBackend(), resolvers=resolvers)
        result = pipe.run_note(note)
        print(result.bundle)
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        *,
        backend: Backend,
        resolvers: ResolverConfig | None = None,
        calibrator: Calibrator | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.config.validate_consistency()
        self.backend = backend
        self.resolvers = resolvers or ResolverConfig()
        self.calibrator = calibrator
        self.schema = build_mention_schema()
        self.schema_digest = schema_digest(self.schema)
        self.cache = CallCache(self.config.cache_path) if self.config.cache_path else None

    # -- one note -------------------------------------------------------

    def run_note(self, note: NoteInput) -> NoteResult:
        """Run the full pipeline on one note."""
        cfg = self.config
        urls = cfg.extension_urls
        result = NoteResult(note=note)

        plan = chunk_note(
            note.text,
            note.note_id,
            max_chars=cfg.max_chars,
            overlap_chars=cfg.overlap_chars,
            section_headers=cfg.section_headers,
            respect_sections=cfg.respect_sections,
        )

        # --- extract ---------------------------------------------------
        raw_items: list[tuple[Mention, Resolution, Grounding, str, int]] = []
        sample_sets: list[list[tuple[Mention, Resolution]]] = []
        total_repairs = 0
        tokens_in = tokens_out = 0
        latency = 0.0
        n_mentions = 0

        for chunk in plan.chunks:
            extraction = extract_chunk(
                chunk,
                self.backend,
                self.schema,
                note_datetime=note.note_datetime.isoformat(),
                temperature=cfg.temperature,
                samples=cfg.samples,
                max_repairs=cfg.max_repairs,
                cache=self.cache,
                schema_digest=self.schema_digest,
            )
            total_repairs += extraction.repairs
            tokens_in += extraction.tokens_in
            tokens_out += extraction.tokens_out
            latency += extraction.latency_s

            if not extraction.ok:
                result.errors.append(
                    f"chunk {chunk.chunk_id} failed after {extraction.repairs} repair(s): "
                    f"{extraction.error}"
                )
                continue

            # Extra samples are kept only to measure consistency.
            for sample in extraction.samples[1:]:
                sample_sets.append(
                    [(m, self.resolvers.resolve(m.text, m.mention_type)) for m in sample.mentions]
                )

            for mention in extraction.mentions.mentions:
                n_mentions += 1
                grounding = ground_mention(
                    mention.evidence,
                    plan.chunks,
                    note.text,
                    preferred_chunk_id=chunk.chunk_id,
                    fuzzy_threshold=cfg.fuzzy_threshold,
                )
                if grounding is None:
                    result.dropped.append((mention, DropReason.UNGROUNDED))
                    result.debug.append(
                        MentionTrace(
                            mention=mention,
                            chunk_id=chunk.chunk_id,
                            drop_reason=DropReason.UNGROUNDED,
                        )
                    )
                    continue
                resolution = self.resolvers.resolve(mention.text, mention.mention_type)
                raw_items.append(
                    (mention, resolution, grounding, chunk.chunk_id, extraction.repairs)
                )

        # --- dedup -----------------------------------------------------
        groups = dedup_mentions([(m, r, g) for m, r, g, _c, _rep in raw_items])
        repairs_by_key = {
            (m.mention_type.value, m.normalized_text, m.assertion.value): rep
            for m, _r, _g, _c, rep in raw_items
        }
        chunk_by_key = {
            (m.mention_type.value, m.normalized_text, m.assertion.value): c
            for m, _r, _g, c, _rep in raw_items
        }

        # --- policy, map, validate, score ------------------------------
        emitted: list[dict[str, Any]] = []
        for index, group in enumerate(groups):
            mention = group.mention
            identity = (
                mention.mention_type.value,
                mention.normalized_text,
                mention.assertion.value,
            )
            chunk_id = chunk_by_key.get(identity, "")
            repairs = repairs_by_key.get(identity, 0)
            best = group.best_grounding
            section = section_at(plan.sections, best.start)

            decision = decide(
                mention,
                emit_refuted=cfg.emit_refuted,
                section=section,
                respect_non_patient_sections=cfg.respect_non_patient_sections,
            )
            trace = MentionTrace(
                mention=mention,
                chunk_id=chunk_id,
                grounding=best,
                resolution=group.resolution,
                decision=decision,
                n_duplicates=group.n_duplicates,
            )

            if not decision.emit:
                reason = decision.reason or DropReason.UNSUPPORTED
                result.dropped.append((mention, reason))
                trace.drop_reason = reason
                result.debug.append(trace)
                continue

            components = self._components(group, repairs, sample_sets, n_samples=cfg.samples)
            score = aggregate(components, cfg.weights)
            calibrated = False
            if self.calibrator is not None and self.calibrator.fitted:
                score = self.calibrator.transform(score)
                calibrated = True

            trace.components = components
            trace.score = score
            trace.calibrated = calibrated

            if score < cfg.min_confidence:
                result.dropped.append((mention, DropReason.BELOW_THRESHOLD))
                trace.drop_reason = DropReason.BELOW_THRESHOLD
                result.debug.append(trace)
                continue

            resource = self._map(group, decision, note, score, calibrated, index, urls)
            if resource is None:
                result.dropped.append((mention, DropReason.UNSUPPORTED))
                trace.drop_reason = DropReason.UNSUPPORTED
                result.debug.append(trace)
                continue

            report = validate_resource(
                resource,
                note.text,
                urls=urls,
                require_quote_match=cfg.include_quotes,
            )
            trace.validation = report
            if not report.ok:
                result.dropped.append((mention, DropReason.VALIDATION_FAILED))
                trace.drop_reason = DropReason.VALIDATION_FAILED
                result.validation.extend(report)
                result.debug.append(trace)
                continue

            result.validation.extend(report)
            trace.resource_id = resource["id"]
            trace.resource_type = resource["resourceType"]
            emitted.append(resource)
            result.debug.append(trace)

        # --- provenance and bundle -------------------------------------
        if emitted:
            from fhirground import __version__
            from fhirground.validate import parse_with_models

            provenance = build_provenance(
                resource_id=f"prov-{note.note_id}",
                target_references=[f"{r['resourceType']}/{r['id']}" for r in emitted],
                source_reference=note.source_reference,
                patient_ref=note.patient_ref,
                model_id=getattr(self.backend, "model_id", "unknown"),
                prompt_version=PROMPT_VERSION,
                package_version=__version__,
                fhir_library=_fhir_library(),
                schema_digest=self.schema_digest,
                backend_name=getattr(self.backend, "name", "unknown"),
            )
            prov_report = parse_with_models(provenance)
            result.validation.extend(prov_report)
            if prov_report.ok:
                emitted.append(provenance)

        result.resources = emitted
        result.bundle = build_bundle(emitted, bundle_id=f"bundle-{note.note_id}")
        result.validation.extend(
            __import__("fhirground.validate", fromlist=["parse_with_models"]).parse_with_models(
                result.bundle
            )
        )

        result.stats = {
            "n_chunks": len(plan.chunks),
            "chunk_coverage": round(plan.coverage(len(note.text)), 6),
            "n_mentions": n_mentions,
            "n_groups": len(groups),
            "n_resources": len([r for r in emitted if r["resourceType"] != "Provenance"]),
            "n_dropped": len(result.dropped),
            "repairs": total_repairs,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "latency_s": round(latency, 4),
            "drop_counts": result.drop_counts(),
            "resource_counts": result.resource_counts(),
        }
        return result

    # -- many notes -----------------------------------------------------

    def run(self, notes: Iterable[NoteInput], concurrency: int = 4) -> Iterator[NoteResult]:
        """Run over many notes.

        Args:
            notes: The notes.
            concurrency: Worker threads. The work is IO-bound on a backend
                call, so threads are the right tool. ``1`` runs inline, which
                keeps tracebacks readable while debugging.

        Yields:
            One :class:`NoteResult` per note, in input order.
        """
        note_list = list(notes)
        if concurrency <= 1:
            for note in note_list:
                yield self.run_note(note)
            return

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            yield from pool.map(self.run_note, note_list)

    # -- internals ------------------------------------------------------

    def _components(
        self,
        group: MentionGroup,
        repairs: int,
        sample_sets: list[list[tuple[Mention, Resolution]]],
        *,
        n_samples: int,
    ) -> ConfidenceComponents:
        """Assemble the confidence components for one group."""
        consistency = None
        if n_samples > 1 and sample_sets:
            matches = matches_for_consistency((group.mention, group.resolution), sample_sets)
            consistency = self_consistency(matches, len(sample_sets))

        terminology = group.resolution.score if group.resolution.is_coded else None
        return ConfidenceComponents(
            g=group.best_grounding.score,
            c=consistency,
            t=terminology,
            v=validation_component(repairs),
            l=None,
        )

    def _map(
        self,
        group: MentionGroup,
        decision: EmissionDecision,
        note: NoteInput,
        score: float,
        calibrated: bool,
        index: int,
        urls: ExtensionUrls,
    ) -> dict[str, Any] | None:
        """Build the resource for one group, then attach every evidence span."""
        cfg = self.config
        mention = group.mention
        best = group.best_grounding
        resource_id = f"{note.note_id}-{mention.mention_type.value[:4]}-{index}"

        if mention.mention_type is MentionType.CONDITION:
            resource = build_condition(
                mention,
                group.resolution,
                best,
                decision,
                note,
                confidence=score,
                calibrated=calibrated,
                resource_id=resource_id,
                include_quotes=cfg.include_quotes,
                include_category=cfg.include_category,
                urls=urls,
            )
            extra = add_condition_evidence
        elif mention.mention_type is MentionType.MEDICATION:
            authored = (
                resolve_date(mention.date_text, note.note_datetime) if mention.date_text else None
            )
            resource = build_medication_request(
                mention,
                group.resolution,
                best,
                decision,
                note,
                confidence=score,
                calibrated=calibrated,
                resource_id=resource_id,
                authored_on=authored,
                include_quotes=cfg.include_quotes,
                urls=urls,
            )
            extra = add_med_evidence
        elif mention.mention_type is MentionType.OBSERVATION:
            effective = (
                resolve_date(mention.date_text, note.note_datetime) if mention.date_text else None
            )
            if effective is None and cfg.infer_observation_effective:
                effective = inferred_from_note(note.note_datetime)
            value = parse_value(
                mention.value_text if mention.value_number is None else mention.value_number
            )
            resource = build_observation(
                mention,
                group.resolution,
                best,
                decision,
                note,
                confidence=score,
                calibrated=calibrated,
                resource_id=resource_id,
                value=value,
                effective=effective,
                include_quotes=cfg.include_quotes,
                mark_inferred_effective=cfg.mark_inferred_effective,
                urls=urls,
            )
            extra = add_obs_evidence
        else:  # pragma: no cover - MentionType is exhaustive
            return None

        # Every merged span, beyond the representative one already attached.
        for grounding in group.groundings:
            if grounding is best:
                continue
            extra(resource, grounding, note, include_quotes=cfg.include_quotes, urls=urls)
        return resource


def _fhir_library() -> str:
    """The FHIR model library and version, recorded in Provenance."""
    try:
        import fhir.resources as fr

        return f"fhir.resources {getattr(fr, '__version__', '?')} (R4B)"
    except ImportError:  # pragma: no cover
        return "fhir.resources (unknown)"


def run_notes(
    notes: Iterable[NoteInput],
    backend: Backend,
    *,
    config: PipelineConfig | None = None,
    resolvers: ResolverConfig | None = None,
    calibrator: Calibrator | None = None,
    concurrency: int = 4,
) -> list[NoteResult]:
    """Convenience wrapper: build a Pipeline and run it to completion."""
    pipe = Pipeline(config, backend=backend, resolvers=resolvers, calibrator=calibrator)
    return list(pipe.run(notes, concurrency=concurrency))


def summarize(results: list[NoteResult], *, started: datetime | None = None) -> dict[str, Any]:
    """Aggregate a run into the ``run_summary.json`` shape (SPEC §6.4)."""
    from fhirground import __version__

    resource_counts: dict[str, int] = {}
    drop_counts: dict[str, int] = {r.value: 0 for r in DropReason}
    tokens_in = tokens_out = repairs = n_mentions = 0
    latency = 0.0
    n_errors = 0

    for result in results:
        for key, value in result.resource_counts().items():
            resource_counts[key] = resource_counts.get(key, 0) + value
        for key, value in result.drop_counts().items():
            drop_counts[key] = drop_counts.get(key, 0) + value
        stats = result.stats
        tokens_in += int(stats.get("tokens_in") or 0)
        tokens_out += int(stats.get("tokens_out") or 0)
        repairs += int(stats.get("repairs") or 0)
        n_mentions += int(stats.get("n_mentions") or 0)
        latency += float(stats.get("latency_s") or 0.0)
        n_errors += len(result.errors)

    return {
        "n_notes": len(results),
        "n_notes_with_errors": sum(1 for r in results if r.errors),
        "n_mentions": n_mentions,
        "resource_counts": resource_counts,
        "drop_counts": drop_counts,
        "repairs": repairs,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "latency_s": round(latency, 4),
        "n_validation_errors": sum(len(r.validation.errors) for r in results),
        "n_validation_warnings": sum(len(r.validation.warnings) for r in results),
        "n_errors": n_errors,
        "versions": {
            "fhirground": __version__,
            "prompt": PROMPT_VERSION,
            "fhir_library": _fhir_library(),
        },
        "started": started.isoformat() if started else None,
    }
