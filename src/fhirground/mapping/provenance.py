"""One ``Provenance`` resource per note, linking outputs to their source.

This is what makes a Bundle auditable six months later. Without it, a reader
has resources with evidence offsets but no record of *which model, which
prompt, and which code version* produced them -- so a disagreement between two
runs is impossible to explain.

``Provenance.agent`` carries the model as a ``Device`` with a display string
naming the model id and the package version. The FHIR ``Device`` is not a real
resource in the Bundle: ``agent.who`` uses ``display`` only, because inventing
a ``Device/<id>`` reference that resolves to nothing would be worse than a
descriptive string.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

#: Provenance activity code: this resource was derived from a source document.
ACTIVITY_SYSTEM = "http://terminology.hl7.org/CodeSystem/v3-DataOperation"
#: Agent type: the software that performed the derivation.
AGENT_TYPE_SYSTEM = "http://terminology.hl7.org/CodeSystem/provenance-participant-type"


def build_provenance(
    *,
    resource_id: str,
    target_references: list[str],
    source_reference: str | None,
    patient_ref: str,
    model_id: str,
    prompt_version: str,
    package_version: str,
    fhir_library: str,
    schema_digest: str | None = None,
    recorded: datetime | None = None,
    backend_name: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the per-note ``Provenance``.

    Args:
        resource_id: Deterministic id.
        target_references: ``ResourceType/id`` for every resource this note
            produced. A Provenance with no targets is not emitted by the
            pipeline, since it would assert nothing.
        source_reference: The source ``DocumentReference``, when known.
        patient_ref: Subject of the note, recorded so Provenance can be found
            by patient.
        model_id: The model that produced the mentions.
        prompt_version: Version string of the prompt template used.
        package_version: fhirground's version.
        fhir_library: The FHIR model library and version, e.g.
            ``fhir.resources 8.3.0 (R4B)``.
        schema_digest: Short hash of the mention schema the model decoded
            against, so two runs with different schemas are distinguishable.
        recorded: Timestamp. Defaults to now, in UTC.
        backend_name: Which backend ran, e.g. ``mock`` or ``bedrock``.
        extra: Additional key/value pairs to record as agent extensions.

    Returns:
        A ``Provenance`` resource dict.

    Raises:
        ValueError: If there are no targets.
    """
    if not target_references:
        raise ValueError(
            "a Provenance needs at least one target; a note that produced no "
            "resources should produce no Provenance either"
        )

    when = (recorded or datetime.now(UTC)).isoformat()

    agent_label_parts = [f"fhirground {package_version}"]
    if backend_name:
        agent_label_parts.append(f"backend={backend_name}")
    agent_label_parts.append(f"model={model_id}")
    agent_label_parts.append(f"prompt={prompt_version}")
    if schema_digest:
        agent_label_parts.append(f"schema={schema_digest}")
    agent_label_parts.append(fhir_library)

    resource: dict[str, Any] = {
        "resourceType": "Provenance",
        "id": resource_id,
        "target": [{"reference": ref} for ref in target_references],
        "recorded": when,
        "activity": {
            "coding": [{"system": ACTIVITY_SYSTEM, "code": "DERIVE", "display": "derive"}]
        },
        "agent": [
            {
                "type": {
                    "coding": [
                        {
                            "system": AGENT_TYPE_SYSTEM,
                            "code": "assembler",
                            "display": "Assembler",
                        }
                    ]
                },
                "who": {"display": " | ".join(agent_label_parts)},
            }
        ],
    }

    if source_reference:
        resource["entity"] = [{"role": "source", "what": {"reference": source_reference}}]

    if patient_ref:
        resource["agent"][0]["onBehalfOf"] = {"reference": patient_ref}

    if extra:
        resource["agent"][0]["extension"] = [
            {"url": f"urn:fhirground:run:{k}", "valueString": str(v)}
            for k, v in sorted(extra.items())
        ]

    return resource


def build_bundle(
    resources: list[dict[str, Any]], *, bundle_id: str, timestamp: datetime | None = None
) -> dict[str, Any]:
    """Assemble resources into a ``collection`` Bundle.

    ``collection`` rather than ``transaction`` because fhirground does not
    write to a server: a transaction Bundle would imply an intent to POST that
    v1 explicitly does not support.
    """
    when = (timestamp or datetime.now(UTC)).isoformat()
    return {
        "resourceType": "Bundle",
        "id": bundle_id,
        "type": "collection",
        "timestamp": when,
        "entry": [
            {
                "fullUrl": f"urn:uuid:{r['resourceType']}-{r['id']}",
                "resource": r,
            }
            for r in resources
        ],
    }
