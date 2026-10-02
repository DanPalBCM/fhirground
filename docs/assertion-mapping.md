# Assertion and mapping rules

This is the normative table from SPEC §4.4, and the module that implements it
(`fhirground.assertion`) is the one that keeps *"denies chest pain"* from
becoming a Condition.

## The table

| Mention | Assertion | Behaviour |
| --- | --- | --- |
| condition | `present` | `Condition`, `verificationStatus=confirmed`; `clinicalStatus` only from a hint |
| condition | `possible` | `Condition`, `verificationStatus=provisional` |
| condition | `negated` | **dropped**; with `emit_refuted: true` → `verificationStatus=refuted` |
| condition | `historical` | `Condition`, `confirmed`; `clinicalStatus` from the hint (often `resolved`) else omitted |
| any | `family` | **dropped** and counted (FamilyMemberHistory is out of scope) |
| any | `hypothetical` | **dropped** and counted |
| medication | `start` / `continue` / `change` | `MedicationRequest`, `status=active`, `intent=order` |
| medication | `stop` | `MedicationRequest`, `status=stopped`, `intent=order` |
| medication | `mention` | **not emitted** in v1; kept in debug (MedicationStatement is v1.1) |
| medication | `negated` | **dropped** |
| observation | `present` | `Observation`, `status=final`, category `vital-signs` or `laboratory` |
| observation | anything else | **dropped** |

Every row is a test. The policy is a lookup, not a heuristic: a combination not
in the table is dropped with reason `unsupported` rather than guessed at.

## The section override

A finding under a **Family History** heading is family history even when the
sentence carries no cue. "Hypertension." under that heading has nothing for a
model to notice, so the section is better evidence than the sentence:

```python
decide(mention, section="family history")
# -> emit=False, reason=FAMILY
```

Disable with `respect_non_patient_sections=False`.

## Drop reasons

Every dropped mention is counted under one of:

`ungrounded`, `negated`, `family`, `hypothetical`, `med_mention_only`,
`validation_failed`, `below_threshold`, `unsupported`, `non_patient_section`.

These appear in `run_summary.json` and in the CLI output, not buried in a debug
file. A pipeline that drops most of what it finds is not working, and that
should be visible.

## Field mapping

**All resources** get `subject = Reference(patient_ref)` — required on input,
an error if missing — and `encounter` when `encounter_ref` is given. Each
carries exactly one `extraction-confidence` extension and at least one
`evidence` extension.

### Condition

`code` from the resolver, or `code.text` only. `onsetString` from `onset_text`.
`category` omitted by default.

!!! note "Why `clinicalStatus` is usually absent"
    FHIR invariant **con-3** requires `clinicalStatus` only when `category`
    **is** `problem-list-item` — note the polarity. Since fhirground omits
    `category` by default, con-3 does not bind, and defaulting the field to
    `active` would assert a clinical course the note never stated.

    Turning on `include_category` makes con-3 bind, so the mapper then
    **refuses** to emit a Condition with no `clinical_status_hint` rather than
    produce a resource that trips a spec warning.

### MedicationRequest

`medicationCodeableConcept` (the R4/R4B spelling; R5 replaced it with a
`CodeableReference`). `dosageInstruction[0].text` is the **evidence sentence** —
the most faithful thing to put there, since it is what the prescriber wrote.
`doseAndRate[0].doseQuantity` only when dose and unit both parse. `route.text`
from the mention.

!!! warning "`authoredOn` is never inferred"
    It is set **only** from an explicitly resolved date. An inferred
    `ResolvedDate` is rejected by the mapper. `authoredOn` is the date an order
    was written; defaulting it to "when the note was typed" fabricates a
    clinical fact with no flag to mark it. Contrast Observation, where an
    inferred time *is* allowed because the `effective-inferred` extension
    records that it was.

`frequency_text` is appended to the human-readable dosage text rather than
parsed into `timing.repeat`. "BID PRN" does not decompose into a count and a
period reliably, and a wrong `Timing` asserts more precision than the note
supports.

### Observation

`code` from the resolver. `valueQuantity` when a number parsed — with
`comparator` and UCUM `system`/`code` when known — otherwise `valueString`.
`effectiveDateTime` from a resolved date, or from `note_datetime` flagged with
`effective-inferred`.

Exactly one `value[x]` is ever set. A blood-pressure pair like `120/80` stays a
`valueString` **with an `Observation.note`** saying why; splitting it into
invented components would be hallucinating structure. Ranges put the low bound
in `valueQuantity` and record the upper bound in a note.

## The FHIR invariants

Encoded in `validate.check_fhir_invariants` with text quoted verbatim from the
R4 StructureDefinitions, which were fetched and read rather than recalled:

| Key | Severity | Text |
| --- | --- | --- |
| con-3 | warning | "Condition.clinicalStatus SHALL be present if verificationStatus is not entered-in-error and category is problem-list-item" |
| con-4 | error | "If condition is abated, then clinicalStatus must be either inactive, resolved, or remission" |
| con-5 | error | "Condition.clinicalStatus SHALL NOT be present if verification Status is entered-in-error" |
| obs-6 | error | "dataAbsentReason SHALL only be present if Observation.value[x] is not present" |
| obs-7 | error | "If Observation.code is the same as an Observation.component.code then the value element associated with the code SHALL NOT be present" |

The spacing typo in con-5 is the spec's own, reproduced so the text is
searchable against the source. See [decisions](decisions.md) D-005.
