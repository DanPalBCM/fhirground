"""Outputs: Bundles, NDJSON, clinval predictions, debug traces, run summary.

Four formats because four audiences:

* **Bundles** -- one JSON per note, for a FHIR consumer.
* **NDJSON per resource type** -- bulk-data style, for loading into a store.
* **clinval predictions** -- so the sibling validation harness can score this
  output with no translation step. The contract is
  ``label = "<ResourceType>:<code or text>"``.
* **debug JSONL** -- one trace per mention, with grounding, resolution,
  confidence components, and the emitted id or drop reason. This is the file
  that makes a disappointing result diagnosable.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from fhirground.mapping.extensions import ExtensionUrls, read_confidence, read_evidence
from fhirground.pipeline import NoteResult, summarize

#: Resource types written to their own NDJSON file.
NDJSON_TYPES = ("Condition", "MedicationRequest", "Observation", "Provenance")

#: Output format names accepted by ``--format``.
FORMATS = ("bundle", "ndjson", "clinval", "debug", "summary")


def _write_json(obj: Any, path: Path, *, indent: int | None = 2) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=indent, default=str), encoding="utf-8")
    return path


def write_bundles(results: Iterable[NoteResult], outdir: str | Path) -> list[Path]:
    """One Bundle JSON per note, named by note id."""
    target = Path(outdir) / "bundles"
    target.mkdir(parents=True, exist_ok=True)
    written = []
    for result in results:
        if not result.bundle:
            continue
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in result.note.note_id)
        written.append(_write_json(result.bundle, target / f"{safe}.json"))
    return written


def write_ndjson(results: Iterable[NoteResult], outdir: str | Path) -> dict[str, Path]:
    """One NDJSON file per resource type, bulk-data style.

    A type with no resources still gets an **empty file**, so a consumer can
    tell "none found" from "the run did not produce this type".
    """
    target = Path(outdir) / "ndjson"
    target.mkdir(parents=True, exist_ok=True)

    buckets: dict[str, list[dict[str, Any]]] = {t: [] for t in NDJSON_TYPES}
    for result in results:
        for resource in result.resources:
            buckets.setdefault(str(resource.get("resourceType")), []).append(resource)

    out: dict[str, Path] = {}
    for resource_type, resources in buckets.items():
        path = target / f"{resource_type}.ndjson"
        with path.open("w", encoding="utf-8") as fh:
            for resource in resources:
                fh.write(json.dumps(resource, default=str))
                fh.write("\n")
        out[resource_type] = path
    return out


def clinval_label(resource: dict[str, Any]) -> str:
    """The clinval label for a resource: ``<ResourceType>:<code or text>``.

    The contract clinval expects. It treats the label as an opaque string and
    never splits it, so a colon in the code is safe.
    """
    resource_type = str(resource.get("resourceType"))
    concept = resource.get("code") or resource.get("medicationCodeableConcept") or {}
    codings = concept.get("coding") or []
    if codings and codings[0].get("code"):
        return f"{resource_type}:{codings[0]['code']}"
    text = (concept.get("text") or "").strip()
    normalized = " ".join(text.lower().split()) or "unknown"
    return f"{resource_type}:{normalized}"


def write_clinval_predictions(
    results: Iterable[NoteResult],
    outdir: str | Path,
    *,
    system_name: str = "fhirground",
    model_id: str = "",
    prompt_version: str = "",
    urls: ExtensionUrls | None = None,
) -> Path:
    """Write ``clinval_predictions.jsonl``.

    Every emitted resource becomes one prediction record with ``value=true``.
    Dropped mentions are **not** emitted as ``value=false``: a drop means
    fhirground declined to assert the finding, which is not the same as
    asserting its absence, and clinval reads a missing record as *unscored*
    rather than as a negative. Materializing negatives needs the label
    universe, which only the study knows.
    """
    target = Path(outdir)
    target.mkdir(parents=True, exist_ok=True)
    path = target / "clinval_predictions.jsonl"

    with path.open("w", encoding="utf-8") as fh:
        for result in results:
            note = result.note
            stats = result.stats
            run_meta: dict[str, Any] = {
                "model": model_id or "unknown",
                "prompt_version": prompt_version or "unknown",
                # One pipeline pass over a note produces every record for that
                # note, so they share a call id and clinval charges the cost
                # once rather than per label.
                "call_id": note.note_id,
            }
            if stats.get("tokens_in"):
                run_meta["tokens_in"] = int(stats["tokens_in"])
            if stats.get("tokens_out"):
                run_meta["tokens_out"] = int(stats["tokens_out"])
            if stats.get("latency_s"):
                run_meta["latency_s"] = float(stats["latency_s"])

            for resource in result.resources:
                if resource.get("resourceType") == "Provenance":
                    continue
                record: dict[str, Any] = {
                    "system": system_name,
                    "note_id": note.note_id,
                    "patient_id": note.patient_ref,
                    "label": clinval_label(resource),
                    "value": True,
                    "note_datetime": note.note_datetime.isoformat(),
                    "run": run_meta,
                }
                confidence = read_confidence(resource, urls)
                if confidence is not None:
                    record["confidence"] = confidence
                evidence = read_evidence(resource, urls)
                if evidence and evidence[0].get("start") is not None:
                    span: dict[str, Any] = {
                        "start": evidence[0]["start"],
                        "end": evidence[0]["end"],
                    }
                    if evidence[0].get("quote"):
                        span["text"] = evidence[0]["quote"]
                    record["span"] = span
                fh.write(json.dumps(record, default=str))
                fh.write("\n")
    return path


def write_debug(results: Iterable[NoteResult], outdir: str | Path) -> Path:
    """One debug trace per mention, as JSONL."""
    target = Path(outdir)
    target.mkdir(parents=True, exist_ok=True)
    path = target / "debug.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for result in results:
            for trace in result.debug:
                row = trace.as_dict()
                row["note_id"] = result.note.note_id
                fh.write(json.dumps(row, default=str))
                fh.write("\n")
    return path


def write_summary(
    results: list[NoteResult],
    outdir: str | Path,
    *,
    started: datetime | None = None,
    backend_description: dict[str, Any] | None = None,
    resolver_description: dict[str, str] | None = None,
    cost_per_1k: dict[str, float] | None = None,
    cache_stats: dict[str, int] | None = None,
) -> Path:
    """Write ``run_summary.json``."""
    summary = summarize(results, started=started)
    if backend_description:
        summary["backend"] = backend_description
    if resolver_description:
        summary["resolvers"] = resolver_description
    if cache_stats:
        summary["cache"] = cache_stats

    if cost_per_1k:
        tokens_in = summary.get("tokens_in") or 0
        tokens_out = summary.get("tokens_out") or 0
        if tokens_in or tokens_out:
            summary["cost_estimate"] = {
                "currency": "USD",
                "input_per_1k": cost_per_1k.get("input_per_1k", 0.0),
                "output_per_1k": cost_per_1k.get("output_per_1k", 0.0),
                "total": round(
                    tokens_in / 1000 * cost_per_1k.get("input_per_1k", 0.0)
                    + tokens_out / 1000 * cost_per_1k.get("output_per_1k", 0.0),
                    6,
                ),
            }
        else:
            summary["cost_estimate"] = {
                "status": "unknown",
                "reason": "the backend reported no token counts",
            }

    return _write_json(summary, Path(outdir) / "run_summary.json")


def write_outputs(
    results: list[NoteResult],
    outdir: str | Path,
    *,
    formats: Iterable[str] = FORMATS,
    system_name: str = "fhirground",
    model_id: str = "",
    prompt_version: str = "",
    started: datetime | None = None,
    backend_description: dict[str, Any] | None = None,
    resolver_description: dict[str, str] | None = None,
    cost_per_1k: dict[str, float] | None = None,
    cache_stats: dict[str, int] | None = None,
    urls: ExtensionUrls | None = None,
) -> dict[str, Any]:
    """Write every requested output format.

    Returns:
        A mapping of format name to the path or paths written.

    Raises:
        ValueError: On an unknown format name, rather than silently writing
            nothing the caller asked for.
    """
    wanted = [f.strip() for f in formats if f and f.strip()]
    unknown = [f for f in wanted if f not in FORMATS]
    if unknown:
        raise ValueError(f"unknown output format(s) {unknown}; known: {list(FORMATS)}")

    target = Path(outdir)
    target.mkdir(parents=True, exist_ok=True)
    written: dict[str, Any] = {}

    if "bundle" in wanted:
        written["bundles"] = write_bundles(results, target)
    if "ndjson" in wanted:
        written["ndjson"] = write_ndjson(results, target)
    if "clinval" in wanted:
        written["clinval"] = write_clinval_predictions(
            results,
            target,
            system_name=system_name,
            model_id=model_id,
            prompt_version=prompt_version,
            urls=urls,
        )
    if "debug" in wanted:
        written["debug"] = write_debug(results, target)
    if "summary" in wanted:
        written["summary"] = write_summary(
            results,
            target,
            started=started,
            backend_description=backend_description,
            resolver_description=resolver_description,
            cost_per_1k=cost_per_1k,
            cache_stats=cache_stats,
        )
    return written
