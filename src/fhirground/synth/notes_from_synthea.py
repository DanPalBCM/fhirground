"""Building the synthetic benchmark, from Synthea output or from nothing.

Two sources of structured records, one renderer:

* :func:`records_from_synthea` reads real Synthea FHIR R4 bundles.
* :func:`generate_records` builds equivalent records with no external tool.

The second exists because Synthea is a **Java** program and fhirground's test
and CI environments have no JVM. Rather than ship fixtures nobody can
regenerate, or skip the benchmark, fhirground generates records of the same
shape and ``scripts/regenerate_synthea.sh`` documents the exact Synthea version
and arguments for anyone who does have Java. See ``docs/decisions.md`` D-006.

Either way the output is the same: notes with exact gold FHIR, derived from the
same records, plus value sets covering exactly the codes used.
"""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fhirground.exceptions import InputError
from fhirground.io import NoteInput, write_notes_jsonl
from fhirground.synth.templates import (
    CONDITION_BANK,
    MEDICATION_BANK,
    OBSERVATION_BANK,
    RenderedNote,
    SourceRecord,
    make_source_record,
    render_note,
    value_sets_from_records,
)

#: Epoch for generated timestamps, so output is reproducible.
EPOCH = datetime(2024, 1, 8, 9, 0, tzinfo=UTC)

#: The Synthea release the regeneration script pins.
SYNTHEA_VERSION = "3.3.0"


def generate_records(
    n_patients: int, *, seed: int = 0, base: datetime | None = None
) -> list[SourceRecord]:
    """Generate source records without Synthea.

    Deterministic: the same seed gives byte-identical output.
    """
    if n_patients < 1:
        raise ValueError(f"n_patients must be at least 1, got {n_patients}")
    rng = random.Random(seed)
    anchor = (base or EPOCH).replace(tzinfo=None)
    return [make_source_record(i + 1, rng, base=anchor) for i in range(n_patients)]


def _iter_resources(doc: Any) -> list[dict[str, Any]]:
    """Flatten a Synthea file into a resource list."""
    out: list[dict[str, Any]] = []
    if isinstance(doc, list):
        for item in doc:
            out.extend(_iter_resources(item))
        return out
    if not isinstance(doc, dict):
        return out
    if doc.get("resourceType") == "Bundle":
        for entry in doc.get("entry") or []:
            resource = entry.get("resource") if isinstance(entry, dict) else None
            if isinstance(resource, dict):
                out.extend(_iter_resources(resource))
        return out
    if "resourceType" in doc:
        out.append(doc)
    return out


def _concept_bits(concept: dict[str, Any]) -> tuple[str | None, str | None]:
    """First coding's code and display from a CodeableConcept."""
    for coding in concept.get("coding") or []:
        if isinstance(coding, dict) and coding.get("code"):
            return str(coding["code"]), coding.get("display") or concept.get("text")
    return None, concept.get("text")


def records_from_synthea(path: str | Path, *, limit: int | None = None) -> list[SourceRecord]:
    """Read source records from Synthea FHIR R4 output.

    Args:
        path: A Synthea ``fhir/`` directory or a single bundle file.
        limit: Keep at most this many patients.

    Returns:
        One :class:`SourceRecord` per patient that has at least one usable
        Condition, MedicationRequest, or Observation.

    Raises:
        InputError: If nothing usable was found, with a message naming what
            was searched -- a silent empty list here would look like a
            successful run that produced no benchmark.
    """
    source = Path(path)
    if not source.exists():
        raise InputError("Synthea path not found", path=str(source))

    files = sorted(source.glob("**/*.json")) if source.is_dir() else [source]
    by_patient: dict[str, SourceRecord] = {}

    for file in files:
        try:
            doc = json.loads(file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise InputError(f"malformed JSON: {exc.msg}", path=str(file)) from exc

        for resource in _iter_resources(doc):
            resource_type = resource.get("resourceType")
            subject = (resource.get("subject") or resource.get("patient") or {}).get("reference")
            if resource_type == "Patient":
                patient_id = str(resource.get("id") or "")
                if patient_id:
                    by_patient.setdefault(
                        patient_id,
                        SourceRecord(
                            patient_id=patient_id,
                            note_datetime=EPOCH.replace(tzinfo=None),
                        ),
                    )
                continue
            if not subject or resource_type not in (
                "Condition",
                "MedicationRequest",
                "Observation",
            ):
                continue

            patient_id = str(subject).split("/")[-1].replace("urn:uuid:", "")
            record = by_patient.setdefault(
                patient_id,
                SourceRecord(patient_id=patient_id, note_datetime=EPOCH.replace(tzinfo=None)),
            )

            if resource_type == "Condition":
                code, display = _concept_bits(resource.get("code") or {})
                if code and display:
                    record.conditions.append(
                        {"code": code, "display": display, "abbrev": (display,)}
                    )
            elif resource_type == "MedicationRequest":
                concept = resource.get("medicationCodeableConcept") or {}
                code, display = _concept_bits(concept)
                if code and display:
                    record.medications.append(
                        {
                            "code": code,
                            "display": display.split(" ")[0],
                            "dose": 1,
                            "unit": "tablet",
                            "route": "PO",
                        }
                    )
            else:
                code, display = _concept_bits(resource.get("code") or {})
                quantity = resource.get("valueQuantity") or {}
                if code and display and quantity.get("value") is not None:
                    record.observations.append(
                        {
                            "code": code,
                            "display": display,
                            "value": float(quantity["value"]),
                            "unit": quantity.get("unit") or "",
                            "category": "laboratory",
                            "abbrev": (display,),
                        }
                    )

    usable = [r for r in by_patient.values() if r.conditions or r.medications or r.observations]
    if not usable:
        raise InputError(
            f"found no Patient with a Condition, MedicationRequest, or Observation in "
            f"{len(files)} file(s). Point --synthea-dir at Synthea's `fhir/` output "
            "directory, or omit it to generate records instead",
            path=str(source),
        )
    usable.sort(key=lambda r: r.patient_id)
    return usable[:limit] if limit else usable


def build_benchmark(
    *,
    n_notes: int = 50,
    seed: int = 0,
    synthea_dir: str | Path | None = None,
    inject_distractors: bool = True,
) -> tuple[list[NoteInput], list[RenderedNote]]:
    """Build notes and gold for the synthetic benchmark.

    Args:
        n_notes: How many notes to render.
        seed: Seed for all randomness.
        synthea_dir: Read records from Synthea output instead of generating.
        inject_distractors: Add negated, family-history, and hedged mentions.

    Returns:
        ``(note_inputs, rendered)`` where ``rendered`` carries the gold and the
        distractor manifest.
    """
    rng = random.Random(seed)
    if synthea_dir:
        records = records_from_synthea(synthea_dir, limit=n_notes)
    else:
        records = generate_records(n_notes, seed=seed)

    notes: list[NoteInput] = []
    rendered: list[RenderedNote] = []
    for i, record in enumerate(records[:n_notes]):
        note = render_note(record, i + 1, rng, inject_distractors=inject_distractors)
        rendered.append(note)
        notes.append(
            NoteInput(
                note_id=note.note_id,
                patient_ref=note.patient_ref,
                note_datetime=note.note_datetime,
                text=note.text,
                document_reference_id=f"docref-{note.note_id}",
            )
        )
    return notes, rendered


def write_benchmark(
    outdir: str | Path,
    *,
    n_notes: int = 50,
    seed: int = 0,
    synthea_dir: str | Path | None = None,
    inject_distractors: bool = True,
) -> dict[str, Path]:
    """Write the whole benchmark: notes, gold, value sets, and a manifest.

    Returns:
        A mapping of artifact name to path.
    """
    target = Path(outdir)
    target.mkdir(parents=True, exist_ok=True)
    notes, rendered = build_benchmark(
        n_notes=n_notes,
        seed=seed,
        synthea_dir=synthea_dir,
        inject_distractors=inject_distractors,
    )

    written: dict[str, Path] = {}

    notes_path = target / "notes.jsonl"
    write_notes_jsonl(notes, notes_path)
    written["notes"] = notes_path

    gold_dir = target / "gold"
    gold_dir.mkdir(parents=True, exist_ok=True)
    buckets: dict[str, list[dict[str, Any]]] = {
        "Condition": [],
        "MedicationRequest": [],
        "Observation": [],
    }
    for note in rendered:
        for resource in note.gold:
            buckets[str(resource["resourceType"])].append(resource)
    for resource_type, resources in buckets.items():
        path = gold_dir / f"{resource_type}.ndjson"
        with path.open("w", encoding="utf-8") as fh:
            for resource in resources:
                fh.write(json.dumps(resource))
                fh.write("\n")
        written[f"gold_{resource_type}"] = path

    # Value sets, derived from the codes the records actually use.
    records = (
        records_from_synthea(synthea_dir, limit=n_notes)
        if synthea_dir
        else generate_records(n_notes, seed=seed)
    )
    value_sets = value_sets_from_records(records)
    vs_dir = target / "valuesets"
    vs_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in value_sets.items():
        path = vs_dir / f"{name}.csv"
        with path.open("w", encoding="utf-8", newline="") as fh:
            fh.write("system,code,display,synonyms\n")
            for row in rows:
                synonyms = str(row.get("synonyms") or "").replace(",", ";")
                display = str(row["display"]).replace(",", ";")
                fh.write(f"{row['system']},{row['code']},{display},{synonyms}\n")
        written[f"valueset_{name}"] = path

    manifest = {
        "generator": "fhirground.synth",
        "source": "synthea" if synthea_dir else "generated",
        "synthea_version_documented": SYNTHEA_VERSION,
        "n_notes": len(notes),
        "seed": seed,
        "inject_distractors": inject_distractors,
        "n_gold_resources": sum(len(v) for v in buckets.values()),
        "gold_counts": {k: len(v) for k, v in buckets.items()},
        "distractors": [{"note_id": n.note_id, **d} for n in rendered for d in n.distractors],
        "banks": {
            "conditions": len(CONDITION_BANK),
            "medications": len(MEDICATION_BANK),
            "observations": len(OBSERVATION_BANK),
        },
        "note": (
            "Every artifact here is synthetic. No real patient record was involved. "
            "Gold resources are generated from the same source records the notes were "
            "rendered from, so the benchmark is exact by construction."
        ),
    }
    manifest_path = target / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    written["manifest"] = manifest_path
    return written
