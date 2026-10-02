"""Input: notes JSONL and FHIR DocumentReference."""

from __future__ import annotations

import base64
import json

import pytest
from pydantic import ValidationError

from fhirground.exceptions import InputError
from fhirground.io import (
    NoteInput,
    read_notes,
    read_notes_fhir,
    read_notes_jsonl,
    write_notes_jsonl,
)
from tests.conftest import NOTE_DT


def doc_ref(
    note_id="d1", text="Assessment: diabetes.", subject="Patient/1", date="2026-03-14T10:00:00Z"
):
    content = {
        "attachment": {
            "contentType": "text/plain",
            "data": base64.b64encode(text.encode()).decode(),
        }
    }
    out = {"resourceType": "DocumentReference", "id": note_id, "content": [content]}
    if subject:
        out["subject"] = {"reference": subject}
    if date:
        out["date"] = date
    return out


class TestNoteInput:
    def test_valid(self):
        note = NoteInput(note_id="n1", patient_ref="Patient/1", note_datetime=NOTE_DT, text="x")
        assert note.patient_ref == "Patient/1"

    def test_bare_patient_id_rejected(self):
        """A bare id is not a FHIR reference and would not resolve."""
        with pytest.raises(ValueError, match="must be a FHIR reference"):
            NoteInput(note_id="n1", patient_ref="123", note_datetime=NOTE_DT, text="x")

    def test_patient_ref_is_required(self):
        with pytest.raises(ValidationError):
            NoteInput(note_id="n1", note_datetime=NOTE_DT, text="x")

    def test_empty_text_rejected(self):
        with pytest.raises(ValidationError):
            NoteInput(note_id="n1", patient_ref="Patient/1", note_datetime=NOTE_DT, text="")

    def test_unknown_field_rejected(self):
        with pytest.raises(ValidationError):
            NoteInput(
                note_id="n1", patient_ref="Patient/1", note_datetime=NOTE_DT, text="x", txet="typo"
            )

    def test_source_reference_normalizes(self):
        a = NoteInput(
            note_id="n",
            patient_ref="Patient/1",
            note_datetime=NOTE_DT,
            text="x",
            document_reference_id="d1",
        )
        b = NoteInput(
            note_id="n",
            patient_ref="Patient/1",
            note_datetime=NOTE_DT,
            text="x",
            document_reference_id="DocumentReference/d1",
        )
        assert a.source_reference == b.source_reference == "DocumentReference/d1"

    def test_source_reference_none_without_an_id(self):
        note = NoteInput(note_id="n", patient_ref="Patient/1", note_datetime=NOTE_DT, text="x")
        assert note.source_reference is None

    def test_text_hash_is_stable_and_short(self):
        note = NoteInput(note_id="n", patient_ref="Patient/1", note_datetime=NOTE_DT, text="abc")
        assert note.text_hash() == note.text_hash()
        assert len(note.text_hash()) == 16


class TestJsonl:
    def test_round_trip(self, tmp_path):
        notes = [
            NoteInput(
                note_id=f"n{i}", patient_ref="Patient/1", note_datetime=NOTE_DT, text=f"note {i}"
            )
            for i in range(3)
        ]
        path = tmp_path / "notes.jsonl"
        assert write_notes_jsonl(notes, path) == 3
        assert [n.note_id for n in read_notes_jsonl(path)] == ["n0", "n1", "n2"]

    def test_blank_lines_and_comments_skipped(self, tmp_path):
        path = tmp_path / "n.jsonl"
        row = json.dumps(
            {
                "note_id": "n1",
                "patient_ref": "Patient/1",
                "note_datetime": "2026-03-14T10:30:00",
                "text": "x",
            }
        )
        path.write_text(f"{row}\n\n   \n// comment\n")
        assert len(read_notes_jsonl(path)) == 1

    def test_malformed_json_reports_the_line(self, tmp_path):
        path = tmp_path / "n.jsonl"
        row = json.dumps(
            {
                "note_id": "n1",
                "patient_ref": "Patient/1",
                "note_datetime": "2026-03-14T10:30:00",
                "text": "x",
            }
        )
        path.write_text(f"{row}\n{{bad\n")
        with pytest.raises(InputError, match=":2"):
            read_notes_jsonl(path)

    def test_invalid_record_reports_the_field(self, tmp_path):
        path = tmp_path / "n.jsonl"
        path.write_text(json.dumps({"note_id": "n1", "text": "x"}) + "\n")
        with pytest.raises(InputError, match="patient_ref"):
            read_notes_jsonl(path)

    def test_non_object_line(self, tmp_path):
        path = tmp_path / "n.jsonl"
        path.write_text("[1,2]\n")
        with pytest.raises(InputError, match="must be a JSON object"):
            read_notes_jsonl(path)

    def test_duplicate_note_ids_rejected(self, tmp_path):
        path = tmp_path / "n.jsonl"
        row = json.dumps(
            {
                "note_id": "n1",
                "patient_ref": "Patient/1",
                "note_datetime": "2026-03-14T10:30:00",
                "text": "x",
            }
        )
        path.write_text(f"{row}\n{row}\n")
        with pytest.raises(InputError, match="duplicate note_id"):
            read_notes_jsonl(path)

    def test_missing_file(self, tmp_path):
        with pytest.raises(InputError, match="not found"):
            read_notes_jsonl(tmp_path / "nope.jsonl")

    def test_empty_file(self, tmp_path):
        path = tmp_path / "n.jsonl"
        path.write_text("\n\n")
        with pytest.raises(InputError, match="no records"):
            read_notes_jsonl(path)


class TestFhirInput:
    def test_single_document_reference(self, tmp_path):
        path = tmp_path / "d.json"
        path.write_text(json.dumps(doc_ref()))
        notes = read_notes_fhir(path)
        assert len(notes) == 1
        assert notes[0].text == "Assessment: diabetes."
        assert notes[0].patient_ref == "Patient/1"
        assert notes[0].document_reference_id == "d1"

    def test_bundle(self, tmp_path):
        path = tmp_path / "b.json"
        bundle = {
            "resourceType": "Bundle",
            "entry": [
                {"resource": doc_ref("d1")},
                {"resource": doc_ref("d2")},
                {"resource": {"resourceType": "Patient", "id": "p1"}},
            ],
        }
        path.write_text(json.dumps(bundle))
        assert len(read_notes_fhir(path)) == 2

    def test_directory(self, tmp_path):
        (tmp_path / "a.json").write_text(json.dumps(doc_ref("d1")))
        (tmp_path / "b.json").write_text(json.dumps(doc_ref("d2")))
        assert len(read_notes_fhir(tmp_path)) == 2

    def test_non_text_attachments_are_skipped_not_fatal(self, tmp_path):
        """A Synthea bundle is full of attachments fhirground does not read."""
        pdf = doc_ref("d1")
        pdf["content"][0]["attachment"]["contentType"] = "application/pdf"
        path = tmp_path / "b.json"
        path.write_text(
            json.dumps(
                {
                    "resourceType": "Bundle",
                    "entry": [{"resource": pdf}, {"resource": doc_ref("d2")}],
                }
            )
        )
        notes = read_notes_fhir(path)
        assert [n.note_id for n in notes] == ["d2"]

    def test_missing_subject_is_an_error(self, tmp_path):
        path = tmp_path / "d.json"
        path.write_text(json.dumps(doc_ref(subject=None)))
        with pytest.raises(InputError, match=r"no subject\.reference"):
            read_notes_fhir(path)

    def test_missing_date_is_an_error(self, tmp_path):
        """Without it, relative dates in the note cannot be anchored."""
        path = tmp_path / "d.json"
        path.write_text(json.dumps(doc_ref(date=None)))
        with pytest.raises(InputError, match="neither `date`"):
            read_notes_fhir(path)

    def test_context_period_start_is_accepted(self, tmp_path):
        ref = doc_ref(date=None)
        ref["context"] = {"period": {"start": "2026-03-14T10:00:00Z"}}
        path = tmp_path / "d.json"
        path.write_text(json.dumps(ref))
        assert len(read_notes_fhir(path)) == 1

    def test_encounter_is_carried(self, tmp_path):
        ref = doc_ref()
        ref["context"] = {"encounter": [{"reference": "Encounter/e1"}]}
        path = tmp_path / "d.json"
        path.write_text(json.dumps(ref))
        assert read_notes_fhir(path)[0].encounter_ref == "Encounter/e1"

    def test_bad_base64_is_reported(self, tmp_path):
        ref = doc_ref()
        ref["content"][0]["attachment"]["data"] = "!!!not base64!!!"
        path = tmp_path / "d.json"
        path.write_text(json.dumps(ref))
        with pytest.raises(InputError, match="base64"):
            read_notes_fhir(path)

    def test_no_usable_documents_explains_what_to_do(self, tmp_path):
        path = tmp_path / "d.json"
        path.write_text(json.dumps({"resourceType": "Patient", "id": "p"}))
        with pytest.raises(InputError, match="--input-format jsonl"):
            read_notes_fhir(path)

    def test_missing_path(self, tmp_path):
        with pytest.raises(InputError, match="not found"):
            read_notes_fhir(tmp_path / "nope")

    def test_empty_directory(self, tmp_path):
        with pytest.raises(InputError, match=r"no \.json files"):
            read_notes_fhir(tmp_path)

    def test_malformed_json(self, tmp_path):
        path = tmp_path / "d.json"
        path.write_text("{oops")
        with pytest.raises(InputError, match="malformed JSON"):
            read_notes_fhir(path)


class TestReadNotesDispatch:
    def test_jsonl(self, tmp_path):
        path = tmp_path / "n.jsonl"
        path.write_text(
            json.dumps(
                {
                    "note_id": "n1",
                    "patient_ref": "Patient/1",
                    "note_datetime": "2026-03-14T10:30:00",
                    "text": "x",
                }
            )
            + "\n"
        )
        assert len(read_notes(path, "jsonl")) == 1

    def test_fhir(self, tmp_path):
        path = tmp_path / "d.json"
        path.write_text(json.dumps(doc_ref()))
        assert len(read_notes(path, "fhir")) == 1

    def test_unknown_format(self, tmp_path):
        with pytest.raises(InputError, match="unknown input format"):
            read_notes(tmp_path, "xml")
