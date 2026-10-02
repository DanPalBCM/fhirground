"""Input: notes from JSONL, or from FHIR ``DocumentReference`` resources.

Two input shapes, one model. ``NoteInput`` is what the pipeline consumes;
everything here exists to produce one.

``patient_ref`` is **required**, not optional. Every resource fhirground emits
needs a ``subject``, and a FHIR resource without a subject is not useful to
anyone. Failing at read time with a clear message beats discovering it during
mapping.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Iterable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from fhirground.exceptions import InputError

#: MIME types whose attachment data fhirground knows how to decode.
SUPPORTED_ATTACHMENT_TYPES = ("text/plain",)


class NoteInput(BaseModel):
    """One clinical note plus the metadata needed to emit FHIR from it."""

    model_config = ConfigDict(extra="forbid")

    note_id: str = Field(min_length=1)
    patient_ref: str = Field(
        min_length=1,
        description="FHIR reference such as 'Patient/123'. Required: every emitted "
        "resource needs a subject.",
    )
    note_datetime: datetime = Field(
        description="When the note was written. Anchors relative date resolution."
    )
    text: str = Field(min_length=1)
    encounter_ref: str | None = None
    document_reference_id: str | None = Field(
        default=None,
        description="Id of the source DocumentReference, referenced by evidence extensions.",
    )

    @field_validator("note_id", "patient_ref", mode="before")
    @classmethod
    def _strip(cls, v: object) -> object:
        return v.strip() if isinstance(v, str) else v

    @field_validator("patient_ref")
    @classmethod
    def _check_reference_shape(cls, v: str) -> str:
        """Warn-by-raising on a bare id.

        ``123`` is not a FHIR reference and would produce a resource whose
        subject does not resolve. Requiring ``ResourceType/id`` here is
        cheaper than debugging a Bundle later.
        """
        if "/" not in v:
            raise ValueError(f"patient_ref must be a FHIR reference like 'Patient/{v}', got {v!r}")
        return v

    @property
    def source_reference(self) -> str | None:
        """Reference to the source DocumentReference, if one was given."""
        if self.document_reference_id is None:
            return None
        if "/" in self.document_reference_id:
            return self.document_reference_id
        return f"DocumentReference/{self.document_reference_id}"

    def text_hash(self) -> str:
        """SHA-256 of the note text, for logs that must not contain the text."""
        import hashlib

        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]


def read_notes_jsonl(path: str | Path) -> list[NoteInput]:
    """Read notes from JSONL, one object per line.

    Raises:
        InputError: On a missing file, malformed JSON, or a record that fails
            validation. The message carries the path and 1-based line number.
    """
    path = Path(path)
    if not path.exists():
        raise InputError("notes file not found", path=str(path))

    out: list[NoteInput] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            stripped = raw.strip()
            if not stripped or stripped.startswith("//"):
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise InputError(f"malformed JSON: {exc.msg}", path=str(path), line=lineno) from exc
            if not isinstance(obj, dict):
                raise InputError(
                    f"each line must be a JSON object, got {type(obj).__name__}",
                    path=str(path),
                    line=lineno,
                )
            try:
                note = NoteInput.model_validate(obj)
            except Exception as exc:
                raise InputError(
                    f"invalid note: {_pydantic_detail(exc)}", path=str(path), line=lineno
                ) from exc
            if note.note_id in seen:
                raise InputError(
                    f"duplicate note_id {note.note_id!r}; note ids must be unique",
                    path=str(path),
                    line=lineno,
                )
            seen.add(note.note_id)
            out.append(note)

    if not out:
        raise InputError("notes file contained no records", path=str(path))
    return out


def _pydantic_detail(exc: Exception) -> str:
    """Render a pydantic ValidationError compactly, or fall back to str()."""
    errors = getattr(exc, "errors", None)
    if not callable(errors):
        return str(exc)
    try:
        return "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}" for e in errors()
        )
    except Exception:  # pragma: no cover - defensive
        return str(exc)


def _decode_attachment(attachment: dict[str, Any], *, note_id: str) -> str | None:
    """Decode a ``text/plain`` attachment, or return None if unusable.

    Returns None rather than raising for an unsupported content type, because a
    Synthea bundle legitimately contains attachments fhirground does not read,
    and the caller counts the skips.
    """
    content_type = (attachment.get("contentType") or "").split(";")[0].strip()
    if content_type and content_type not in SUPPORTED_ATTACHMENT_TYPES:
        return None
    data = attachment.get("data")
    if not data:
        return None
    try:
        return base64.b64decode(data).decode("utf-8")
    except Exception as exc:
        raise InputError(
            f"attachment data is not base64-encoded UTF-8: {exc}", note_id=note_id
        ) from exc


def _note_from_document_reference(
    doc: dict[str, Any], *, path: str | None = None
) -> NoteInput | None:
    """Build a NoteInput from a FHIR DocumentReference, or None if unusable."""
    doc_id = str(doc.get("id") or "")
    subject = (doc.get("subject") or {}).get("reference")
    if not subject:
        raise InputError(
            "DocumentReference has no subject.reference, so the emitted resources "
            "would have no patient",
            path=path,
            note_id=doc_id or None,
        )

    text: str | None = None
    for content in doc.get("content") or []:
        attachment = content.get("attachment") or {}
        text = _decode_attachment(attachment, note_id=doc_id)
        if text:
            break
    if not text or not text.strip():
        return None

    when = doc.get("date") or (doc.get("context") or {}).get("period", {}).get("start")
    if not when:
        raise InputError(
            "DocumentReference has neither `date` nor `context.period.start`, so "
            "relative dates in the note could not be anchored",
            path=path,
            note_id=doc_id or None,
        )

    encounter = None
    encounters = (doc.get("context") or {}).get("encounter") or []
    if isinstance(encounters, list) and encounters:
        encounter = encounters[0].get("reference")
    elif isinstance(encounters, dict):
        encounter = encounters.get("reference")

    return NoteInput(
        note_id=doc_id or f"docref-{abs(hash(text)) % 10**8}",
        patient_ref=subject,
        note_datetime=when,  # type: ignore[arg-type]
        text=text,
        encounter_ref=encounter,
        document_reference_id=doc_id or None,
    )


def read_notes_fhir(path: str | Path) -> list[NoteInput]:
    """Read notes from FHIR ``DocumentReference`` resources or Bundles.

    Accepts a single JSON file or a directory of them, which is the shape
    Synthea writes. Resources that are not DocumentReferences, and
    DocumentReferences with no readable ``text/plain`` attachment, are skipped
    rather than treated as errors -- a Synthea bundle is full of both.

    Raises:
        InputError: If nothing usable was found, or a DocumentReference is
            missing the subject or date that makes it usable at all.
    """
    path = Path(path)
    if not path.exists():
        raise InputError("FHIR input path not found", path=str(path))

    files = sorted(path.glob("**/*.json")) if path.is_dir() else [path]
    if not files:
        raise InputError("no .json files found", path=str(path))

    out: list[NoteInput] = []
    seen: set[str] = set()
    for file in files:
        try:
            doc = json.loads(file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise InputError(f"malformed JSON: {exc.msg}", path=str(file)) from exc

        for resource in _iter_resources(doc):
            if resource.get("resourceType") != "DocumentReference":
                continue
            note = _note_from_document_reference(resource, path=str(file))
            if note is None:
                continue
            if note.note_id in seen:
                continue
            seen.add(note.note_id)
            out.append(note)

    if not out:
        raise InputError(
            "no DocumentReference with a readable text/plain attachment was found. "
            f"Looked in {len(files)} file(s). Synthea must be run with text notes "
            "enabled, or use --input-format jsonl instead",
            path=str(path),
        )
    return out


def _iter_resources(doc: Any) -> Iterator[dict[str, Any]]:
    """Yield resources from a bare resource, a Bundle, or a list of either."""
    if isinstance(doc, list):
        for item in doc:
            yield from _iter_resources(item)
        return
    if not isinstance(doc, dict):
        return
    if doc.get("resourceType") == "Bundle":
        for entry in doc.get("entry") or []:
            resource = entry.get("resource") if isinstance(entry, dict) else None
            if isinstance(resource, dict):
                yield from _iter_resources(resource)
        return
    if "resourceType" in doc:
        yield doc


def read_notes(path: str | Path, input_format: str = "jsonl") -> list[NoteInput]:
    """Read notes in either supported format.

    Args:
        path: File or directory.
        input_format: ``"jsonl"`` or ``"fhir"``.
    """
    if input_format == "jsonl":
        return read_notes_jsonl(path)
    if input_format == "fhir":
        return read_notes_fhir(path)
    raise InputError(f"unknown input format {input_format!r}; use 'jsonl' or 'fhir'")


def write_notes_jsonl(notes: Iterable[NoteInput], path: str | Path) -> int:
    """Write notes as JSONL. Returns the number written."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with target.open("w", encoding="utf-8") as fh:
        for note in notes:
            fh.write(note.model_dump_json(exclude_none=True))
            fh.write("\n")
            n += 1
    return n
