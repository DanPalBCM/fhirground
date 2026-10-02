"""Validation: model parsing, project rules, and an optional HL7 validator hook.

Three layers, in increasing cost:

1. **Model parse** -- every resource is parsed with the pinned FHIR R4B models.
   A resource that will not parse is a bug in fhirground, not a warning.
2. **Project rules** (SPEC §4.5) -- the invariants FHIR itself does not state,
   plus the three R4 invariants that bind on the resources v1 emits. These are
   what make the output *auditable* rather than merely well-formed: that every
   evidence offset really indexes the note, that no code arrived without a
   resolver, that the UCUM system never appears beside a code fhirground did
   not resolve.
3. **HL7 FHIR Validator** -- optional, run only when the user supplies the jar
   path. Not bundled: it is a large Java artifact with its own release cadence.

Every rule has a stable name so a failure is actionable, and every rule has
positive and negative tests.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from fhirground.mapping.extensions import ExtensionUrls, read_evidence
from fhirground.normalize.units import UCUM_SYSTEM, is_known_ucum_code

#: Resource types v1 emits. Anything else in a Bundle is a pass-through bug.
ALLOWED_RESOURCE_TYPES = frozenset(
    {"Condition", "MedicationRequest", "Observation", "Provenance", "Bundle"}
)

#: Top-level fields each emitted type is allowed to carry (SPEC §4.5's last
#: rule, which guards against accidentally passing model output straight
#: through into a resource).
ALLOWED_FIELDS: dict[str, frozenset[str]] = {
    "Condition": frozenset(
        {
            "resourceType",
            "id",
            "meta",
            "extension",
            "subject",
            "encounter",
            "code",
            "category",
            "clinicalStatus",
            "verificationStatus",
            "onsetString",
            "onsetDateTime",
            "onsetPeriod",
            "recordedDate",
            "note",
        }
    ),
    "MedicationRequest": frozenset(
        {
            "resourceType",
            "id",
            "meta",
            "extension",
            "subject",
            "encounter",
            "status",
            "intent",
            "medicationCodeableConcept",
            "authoredOn",
            "dosageInstruction",
            "note",
            "reasonCode",
        }
    ),
    "Observation": frozenset(
        {
            "resourceType",
            "id",
            "meta",
            "extension",
            "subject",
            "encounter",
            "status",
            "category",
            "code",
            "valueQuantity",
            "valueString",
            "valueCodeableConcept",
            "valueBoolean",
            "valueInteger",
            "effectiveDateTime",
            "effectivePeriod",
            "issued",
            "note",
        }
    ),
    "Provenance": frozenset(
        {
            "resourceType",
            "id",
            "meta",
            "extension",
            "target",
            "recorded",
            "activity",
            "agent",
            "entity",
        }
    ),
}

#: clinicalStatus codes that satisfy R4 invariant con-4.
CON4_ALLOWED = frozenset({"inactive", "resolved", "remission"})


class Severity(StrEnum):
    """Whether a finding blocks emission.

    ``error`` means the resource is dropped. ``warning`` means it is emitted
    and the finding is reported -- used for FHIR invariants the spec itself
    marks as warnings, such as con-3.
    """

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class Finding:
    """One validation problem."""

    rule: str
    severity: Severity
    message: str
    resource_type: str | None = None
    resource_id: str | None = None

    def __str__(self) -> str:
        where = f" [{self.resource_type}/{self.resource_id}]" if self.resource_id else ""
        return f"{self.severity.value}: {self.rule}: {self.message}{where}"


@dataclass
class ValidationReport:
    """The outcome of validating one or more resources."""

    findings: list[Finding] = field(default_factory=list)

    @property
    def errors(self) -> list[Finding]:
        """Findings that block emission."""
        return [f for f in self.findings if f.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[Finding]:
        """Findings that are reported but do not block."""
        return [f for f in self.findings if f.severity is Severity.WARNING]

    @property
    def ok(self) -> bool:
        """Whether there are no errors."""
        return not self.errors

    def add(
        self,
        rule: str,
        message: str,
        *,
        severity: Severity = Severity.ERROR,
        resource: dict[str, Any] | None = None,
    ) -> None:
        """Record a finding."""
        self.findings.append(
            Finding(
                rule=rule,
                severity=severity,
                message=message,
                resource_type=(resource or {}).get("resourceType"),
                resource_id=(resource or {}).get("id"),
            )
        )

    def extend(self, other: ValidationReport) -> None:
        """Merge another report in."""
        self.findings.extend(other.findings)

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe summary."""
        return {
            "ok": self.ok,
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
            "findings": [
                {
                    "rule": f.rule,
                    "severity": f.severity.value,
                    "message": f.message,
                    "resource_type": f.resource_type,
                    "resource_id": f.resource_id,
                }
                for f in self.findings
            ],
        }


# ---------------------------------------------------------------------------
# layer 1: model parse
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict[str, Any] = {}


def _model_for(resource_type: str) -> Any | None:
    """Import the R4B model class for a resource type, cached."""
    if resource_type in _MODEL_CACHE:
        return _MODEL_CACHE[resource_type]
    module_name = resource_type.lower()
    try:
        module = __import__(f"fhir.resources.R4B.{module_name}", fromlist=[resource_type])
        model = getattr(module, resource_type)
    except (ImportError, AttributeError):  # pragma: no cover - defensive
        return None
    _MODEL_CACHE[resource_type] = model
    return model


def parse_with_models(resource: dict[str, Any]) -> ValidationReport:
    """Parse one resource with the pinned FHIR R4B models."""
    report = ValidationReport()
    resource_type = resource.get("resourceType")
    if not resource_type:
        report.add("model-parse", "resource has no resourceType", resource=resource)
        return report

    model = _model_for(str(resource_type))
    if model is None:
        report.add(
            "model-parse",
            f"no R4B model is available for resourceType {resource_type!r}",
            resource=resource,
        )
        return report

    try:
        model.model_validate(resource)
    except Exception as exc:
        report.add(
            "model-parse",
            f"does not parse as FHIR R4B {resource_type}: {_compact(exc)}",
            resource=resource,
        )
    return report


def _compact(exc: Exception) -> str:
    """One-line rendering of a pydantic error."""
    errors = getattr(exc, "errors", None)
    if callable(errors):
        try:
            return "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}"
                for e in errors()[:4]
            )
        except Exception:  # pragma: no cover
            pass
    return str(exc).replace("\n", " ")[:300]


# ---------------------------------------------------------------------------
# layer 2: project rules
# ---------------------------------------------------------------------------


def check_extensions(
    resource: dict[str, Any], urls: ExtensionUrls | None = None
) -> ValidationReport:
    """Exactly one confidence extension and at least one evidence extension."""
    u = urls or ExtensionUrls()
    report = ValidationReport()
    if resource.get("resourceType") == "Provenance":
        return report

    extensions = resource.get("extension") or []
    n_conf = sum(1 for e in extensions if e.get("url") == u.confidence)
    n_evid = sum(1 for e in extensions if e.get("url") == u.evidence)

    if n_conf != 1:
        report.add(
            "exactly-one-confidence",
            f"expected exactly 1 extraction-confidence extension, found {n_conf}",
            resource=resource,
        )
    if n_evid < 1:
        report.add(
            "at-least-one-evidence",
            "every emitted resource must carry at least one evidence extension; "
            "an ungrounded resource is not auditable",
            resource=resource,
        )
    return report


def check_evidence_offsets(
    resource: dict[str, Any],
    note_text: str,
    *,
    urls: ExtensionUrls | None = None,
    require_quote_match: bool = True,
) -> ValidationReport:
    """Offsets index the note, and the quote matches what is there.

    SPEC §4.5: ``note_text[start:end]`` normalized equals the stored quote
    normalized. When quotes are omitted for PHI minimization the offsets must
    still be in range -- so the claim stays checkable against the note even
    when the resource does not carry the text.
    """
    report = ValidationReport()
    for i, evidence in enumerate(read_evidence(resource, urls)):
        start, end = evidence.get("start"), evidence.get("end")
        if start is None or end is None:
            report.add(
                "evidence-offsets-present",
                f"evidence[{i}] is missing start or end",
                resource=resource,
            )
            continue
        if not (0 <= start <= end <= len(note_text)):
            report.add(
                "evidence-offsets-in-range",
                f"evidence[{i}] offsets [{start}:{end}] are outside the note "
                f"(length {len(note_text)})",
                resource=resource,
            )
            continue
        quote = evidence.get("quote")
        if quote is None:
            continue  # omitted by include_quotes=False; offsets already checked
        if require_quote_match:
            actual = " ".join(note_text[start:end].split()).lower()
            stored = " ".join(quote.split()).lower()
            if actual != stored:
                report.add(
                    "evidence-quote-matches-offsets",
                    f"evidence[{i}] quote does not match note_text[{start}:{end}]",
                    resource=resource,
                )
    return report


def check_codings_have_system_and_code(resource: dict[str, Any]) -> ValidationReport:
    """Every coding has a system and a code.

    SPEC §4.5 also requires the display to have come from a ``Resolution``.
    That half is enforced *structurally* -- the mapping functions accept only
    ``Resolution`` objects -- because no post-hoc inspection of a dict can
    tell where a string came from. This rule catches the observable part.
    """
    report = ValidationReport()
    for path, coding in _iter_codings(resource):
        if not coding.get("system"):
            report.add(
                "coding-has-system",
                f"coding at {path} has no system",
                resource=resource,
            )
        if not coding.get("code"):
            report.add("coding-has-code", f"coding at {path} has no code", resource=resource)
    return report


def check_ucum_usage(resource: dict[str, Any]) -> ValidationReport:
    """The UCUM system appears only beside a code from the bundled table.

    Without this rule, a future change could emit ``http://unitsofmeasure.org``
    beside a guessed code -- a hallucinated code wearing a standards URL, which
    is harder to spot than a plainly missing one.
    """
    report = ValidationReport()
    for path, quantity in _iter_quantities(resource):
        system, code = quantity.get("system"), quantity.get("code")
        if system == UCUM_SYSTEM and not is_known_ucum_code(code):
            report.add(
                "ucum-code-from-table",
                f"{path} declares the UCUM system with code {code!r}, which is not in "
                "fhirground's unit table; an unrecognized unit must be carried as "
                "`unit` text with no system or code",
                resource=resource,
            )
        if code and not system:
            report.add(
                "quantity-code-needs-system",
                f"{path} has a code with no system",
                resource=resource,
            )
    return report


def check_allowed_fields(resource: dict[str, Any]) -> ValidationReport:
    """No field outside the type's allowed set.

    Guards against pass-through: if model output ever leaked into a resource
    dict, it would show up here as an unexpected key.
    """
    report = ValidationReport()
    resource_type = str(resource.get("resourceType") or "")
    allowed = ALLOWED_FIELDS.get(resource_type)
    if allowed is None:
        if resource_type not in ALLOWED_RESOURCE_TYPES:
            report.add(
                "allowed-resource-type",
                f"resourceType {resource_type!r} is not one fhirground emits",
                resource=resource,
            )
        return report
    unexpected = sorted(set(resource) - allowed)
    if unexpected:
        report.add(
            "allowed-fields",
            f"{resource_type} carries unexpected field(s) {unexpected}",
            resource=resource,
        )
    return report


def check_fhir_invariants(resource: dict[str, Any]) -> ValidationReport:
    """The R4 invariants that bind on the resources v1 emits.

    Text quoted verbatim from the R4 StructureDefinitions, verified against
    them rather than recalled (``docs/decisions.md`` D-005).
    """
    report = ValidationReport()
    resource_type = resource.get("resourceType")

    if resource_type == "Condition":
        clinical = _status_codes(resource.get("clinicalStatus"))
        verification = _status_codes(resource.get("verificationStatus"))
        has_clinical = bool(clinical)
        entered_in_error = "entered-in-error" in verification
        categories = _category_codes(resource)

        # con-5 (error): "Condition.clinicalStatus SHALL NOT be present if
        # verification Status is entered-in-error"
        if entered_in_error and has_clinical:
            report.add(
                "con-5",
                "Condition.clinicalStatus SHALL NOT be present if verification Status "
                "is entered-in-error",
                resource=resource,
            )

        # con-4 (error): "If condition is abated, then clinicalStatus must be
        # either inactive, resolved, or remission"
        abated = any(k.startswith("abatement") for k in resource)
        if abated and not (clinical & CON4_ALLOWED):
            report.add(
                "con-4",
                "If condition is abated, then clinicalStatus must be either inactive, "
                f"resolved, or remission (found {sorted(clinical) or 'none'})",
                resource=resource,
            )

        # con-3 (warning): "Condition.clinicalStatus SHALL be present if
        # verificationStatus is not entered-in-error and category is
        # problem-list-item". Note the polarity: it binds only when category
        # *is* problem-list-item, which fhirground omits by default.
        if "problem-list-item" in categories and not entered_in_error and not has_clinical:
            report.add(
                "con-3",
                "Condition.clinicalStatus SHALL be present if verificationStatus is not "
                "entered-in-error and category is problem-list-item",
                severity=Severity.WARNING,
                resource=resource,
            )

    if resource_type == "Observation":
        has_value = any(k.startswith("value") for k in resource)

        # obs-6 (error): "dataAbsentReason SHALL only be present if
        # Observation.value[x] is not present"
        if resource.get("dataAbsentReason") is not None and has_value:
            report.add(
                "obs-6",
                "dataAbsentReason SHALL only be present if Observation.value[x] is not present",
                resource=resource,
            )

        # obs-7 (error): "If Observation.code is the same as an
        # Observation.component.code then the value element associated with the
        # code SHALL NOT be present"
        own = _coding_keys(resource.get("code"))
        for component in resource.get("component") or []:
            if has_value and own & _coding_keys(component.get("code")):
                report.add(
                    "obs-7",
                    "If Observation.code is the same as an Observation.component.code "
                    "then the value element associated with the code SHALL NOT be present",
                    resource=resource,
                )
                break

    return report


def _status_codes(concept: Any) -> set[str]:
    """Codes inside a CodeableConcept, as a set."""
    if not isinstance(concept, dict):
        return set()
    return {
        str(c.get("code"))
        for c in concept.get("coding") or []
        if isinstance(c, dict) and c.get("code")
    }


def _category_codes(resource: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for concept in resource.get("category") or []:
        out |= _status_codes(concept)
    return out


def _coding_keys(concept: Any) -> set[tuple[str, str]]:
    if not isinstance(concept, dict):
        return set()
    return {
        (str(c.get("system")), str(c.get("code")))
        for c in concept.get("coding") or []
        if isinstance(c, dict) and c.get("code")
    }


def _iter_codings(node: Any, path: str = "$") -> Iterable[tuple[str, dict[str, Any]]]:
    """Walk a resource yielding every ``coding`` entry with its path."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "coding" and isinstance(value, list):
                for i, coding in enumerate(value):
                    if isinstance(coding, dict):
                        yield f"{path}.coding[{i}]", coding
            else:
                yield from _iter_codings(value, f"{path}.{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from _iter_codings(value, f"{path}[{i}]")


def _iter_quantities(node: Any, path: str = "$") -> Iterable[tuple[str, dict[str, Any]]]:
    """Walk a resource yielding anything Quantity-shaped."""
    if isinstance(node, dict):
        if "value" in node and ("unit" in node or "code" in node):
            yield path, node
        for key, value in node.items():
            yield from _iter_quantities(value, f"{path}.{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from _iter_quantities(value, f"{path}[{i}]")


def validate_resource(
    resource: dict[str, Any],
    note_text: str | None = None,
    *,
    urls: ExtensionUrls | None = None,
    require_quote_match: bool = True,
) -> ValidationReport:
    """Run the model parse and every project rule on one resource."""
    report = ValidationReport()
    report.extend(parse_with_models(resource))
    report.extend(check_allowed_fields(resource))
    report.extend(check_extensions(resource, urls))
    report.extend(check_codings_have_system_and_code(resource))
    report.extend(check_ucum_usage(resource))
    report.extend(check_fhir_invariants(resource))
    if note_text is not None:
        report.extend(
            check_evidence_offsets(
                resource, note_text, urls=urls, require_quote_match=require_quote_match
            )
        )
    return report


def validate_bundle(
    bundle: dict[str, Any],
    note_text: str | None = None,
    *,
    urls: ExtensionUrls | None = None,
    require_quote_match: bool = True,
) -> ValidationReport:
    """Validate a Bundle and every resource inside it."""
    report = ValidationReport()
    report.extend(parse_with_models(bundle))
    for entry in bundle.get("entry") or []:
        resource = entry.get("resource") if isinstance(entry, dict) else None
        if isinstance(resource, dict):
            report.extend(
                validate_resource(
                    resource,
                    note_text,
                    urls=urls,
                    require_quote_match=require_quote_match,
                )
            )
    return report


# ---------------------------------------------------------------------------
# layer 3: the optional HL7 validator jar
# ---------------------------------------------------------------------------


def run_hl7_validator(
    resources: list[dict[str, Any]],
    jar_path: str | Path,
    *,
    fhir_version: str = "4.0.1",
    java: str | None = None,
    timeout: float = 600.0,
) -> ValidationReport:
    """Run the official HL7 FHIR Validator, if the user supplies the jar.

    The jar is **not bundled**: it is a large Java artifact with its own
    release cadence and licence. This hook shells out to it.

    Args:
        resources: Resources to validate.
        jar_path: Path to ``validator_cli.jar``.
        fhir_version: Version to validate against. ``4.0.1`` is R4; fhirground
            emits R4B-modelled resources, which for these three types are
            field-identical to R4 (``docs/decisions.md`` D-002).
        java: Java executable. Defaults to whatever is on PATH.
        timeout: Seconds before giving up.

    Returns:
        A report. If Java or the jar is missing, that is reported as a single
        finding rather than raised, so an optional hook cannot break a run.
    """
    report = ValidationReport()
    jar = Path(jar_path)
    if not jar.exists():
        report.add("hl7-validator", f"validator jar not found at {jar}")
        return report

    java_bin = java or shutil.which("java")
    if not java_bin:
        report.add(
            "hl7-validator",
            "no `java` on PATH, so the HL7 validator hook was skipped. Install a JRE "
            "or omit --hl7-validator.",
            severity=Severity.WARNING,
        )
        return report

    with tempfile.TemporaryDirectory(prefix="fhirground-hl7-") as tmp:
        tmpdir = Path(tmp)
        paths = []
        for i, resource in enumerate(resources):
            target = tmpdir / f"{resource.get('resourceType', 'Resource')}-{i}.json"
            target.write_text(json.dumps(resource))
            paths.append(str(target))

        cmd = [java_bin, "-jar", str(jar), *paths, "-version", fhir_version]
        try:
            completed = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, check=False
            )
        except (OSError, subprocess.SubprocessError) as exc:
            report.add("hl7-validator", f"could not run the validator: {exc}")
            return report

        output = f"{completed.stdout}\n{completed.stderr}"
        for line in output.splitlines():
            stripped = line.strip()
            lowered = stripped.lower()
            if lowered.startswith("error") or " error " in lowered[:40]:
                report.add("hl7-validator", stripped)
            elif lowered.startswith("warning"):
                report.add("hl7-validator", stripped, severity=Severity.WARNING)

        if completed.returncode != 0 and not report.errors:
            report.add(
                "hl7-validator",
                f"validator exited {completed.returncode} without a parseable error line",
            )
    return report
