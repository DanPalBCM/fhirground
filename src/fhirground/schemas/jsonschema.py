"""Generating the constrained-decoding schema from the Pydantic model.

Pydantic's ``model_json_schema()`` is not usable as a constrained-decoding
target as-is. It emits ``$defs`` with ``$ref`` pointers for every enum, and
expresses optionality as ``anyOf: [{...}, {"type": "null"}]``. SPEC §6.1
requires the schema stay inside the subset that **both** Bedrock tool
``inputSchema`` and vLLM guided decoding accept: flat objects, enums, nullable
fields via type unions, no ``oneOf``/``anyOf`` on objects, no ``$ref``.

So this module takes pydantic's output and:

1. inlines every ``$ref`` from ``$defs`` and deletes ``$defs``;
2. collapses ``anyOf: [{"type": "string"}, {"type": "null"}]`` into
   ``{"type": ["string", "null"]}``;
3. collapses a nullable enum into one ``enum`` list containing ``null``;
4. strips pydantic bookkeeping (``title``, ``default``, ``$schema`` noise) that
   adds nothing for a decoder and differs between pydantic versions;
5. re-checks the result against :func:`assert_backend_safe`, so a future
   pydantic release that changes its output shape fails loudly here rather
   than at a backend.

The generated schema is checked in as ``mention.schema.json`` and a test
asserts the file still matches what the model produces.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from fhirground.schemas.mention import MentionList

#: Where the generated schema is checked in.
SCHEMA_PATH = Path(__file__).parent / "mention.schema.json"

#: Keys that carry no meaning for a constrained decoder and are removed, so the
#: checked-in file does not churn across pydantic releases.
_STRIP_KEYS = ("title", "default", "additionalPropertiesNullable")

#: JSON Schema constructs the common backend subset does not accept.
_FORBIDDEN_KEYS = ("$ref", "$defs", "oneOf", "allOf", "not", "patternProperties", "if", "then")


def _inline_refs(node: Any, defs: dict[str, Any], depth: int = 0) -> Any:
    """Replace ``$ref`` pointers with their definitions, recursively.

    Depth-limited rather than cycle-detecting: the mention schema is flat by
    construction, so a deep nesting means something has gone wrong and should
    surface as an error rather than as a hang.
    """
    if depth > 12:
        raise ValueError(
            "schema nesting exceeded 12 levels while inlining $ref, which suggests a "
            "recursive definition. The mention schema must stay flat (SPEC §6.1)."
        )
    if isinstance(node, dict):
        if "$ref" in node:
            ref = node["$ref"]
            name = ref.rsplit("/", 1)[-1]
            if name not in defs:
                raise ValueError(f"cannot resolve $ref {ref!r}: no definition named {name!r}")
            merged = copy.deepcopy(defs[name])
            # Keep sibling keys (a description on the property, say) but let the
            # definition supply the type information.
            for key, value in node.items():
                if key != "$ref":
                    merged.setdefault(key, value)
            return _inline_refs(merged, defs, depth + 1)
        return {k: _inline_refs(v, defs, depth + 1) for k, v in node.items()}
    if isinstance(node, list):
        return [_inline_refs(v, defs, depth + 1) for v in node]
    return node


def _collapse_nullable(node: Any) -> Any:
    """Turn ``anyOf`` unions with null into type unions or nullable enums."""
    if isinstance(node, list):
        return [_collapse_nullable(v) for v in node]
    if not isinstance(node, dict):
        return node

    node = {k: _collapse_nullable(v) for k, v in node.items()}

    options = node.get("anyOf")
    if not isinstance(options, list):
        return node

    has_null = any(isinstance(o, dict) and o.get("type") == "null" for o in options)
    concrete = [o for o in options if not (isinstance(o, dict) and o.get("type") == "null")]

    # Nullable enum: one enum branch plus null -> a single enum including null.
    # `type` must become a union too, or the declared type contradicts the
    # null member and a strict validator rejects the very value we allowed.
    if has_null and len(concrete) == 1 and isinstance(concrete[0], dict) and "enum" in concrete[0]:
        merged = {k: v for k, v in node.items() if k != "anyOf"}
        merged.update(concrete[0])
        merged["enum"] = [*concrete[0]["enum"], None]
        declared = concrete[0].get("type")
        if isinstance(declared, str):
            merged["type"] = [declared, "null"]
        return merged

    # Nullable scalar: one typed branch plus null -> type union.
    if has_null and len(concrete) == 1 and isinstance(concrete[0], dict) and "type" in concrete[0]:
        merged = {k: v for k, v in node.items() if k != "anyOf"}
        branch = dict(concrete[0])
        branch_type = branch.pop("type")
        merged.update(branch)
        merged["type"] = [branch_type, "null"]
        return merged

    # Single non-null branch (pydantic sometimes wraps a lone option).
    if not has_null and len(concrete) == 1 and isinstance(concrete[0], dict):
        merged = {k: v for k, v in node.items() if k != "anyOf"}
        merged.update(concrete[0])
        return merged

    return node


def _strip_noise(node: Any) -> Any:
    """Remove pydantic bookkeeping keys and shorten descriptions.

    Descriptions are truncated to their first line. Pydantic lifts whole class
    docstrings into ``description``, and this schema is sent to a model on
    every call -- paragraphs of implementation rationale are wasted prompt
    tokens, and they leak internal reasoning into the request. The first line
    is the part written to be read.
    """
    if isinstance(node, list):
        return [_strip_noise(v) for v in node]
    if not isinstance(node, dict):
        return node
    out = {}
    for key, value in node.items():
        if key in _STRIP_KEYS:
            continue
        if key == "description" and isinstance(value, str):
            first = value.strip().split("\n", 1)[0].strip()
            if first:
                out[key] = first
            continue
        out[key] = _strip_noise(value)
    return out


def assert_backend_safe(schema: dict[str, Any], *, path: str = "$") -> None:
    """Raise if ``schema`` leaves the subset both backends accept.

    Checks for the constructs SPEC §6.1 rules out. This is a real guard, not a
    formality: a ``$ref`` that reaches Bedrock produces an opaque validation
    error at request time, far from the cause.

    Raises:
        ValueError: naming the offending key and its JSON path.
    """
    if isinstance(schema, dict):
        for bad in _FORBIDDEN_KEYS:
            if bad in schema:
                raise ValueError(
                    f"schema contains {bad!r} at {path}, which is outside the subset "
                    "accepted by both Bedrock tool inputSchema and vLLM guided decoding "
                    "(SPEC §6.1)"
                )
        if "anyOf" in schema:
            raise ValueError(
                f"schema contains 'anyOf' at {path}; express optionality as a type "
                "union such as {'type': ['string', 'null']} instead"
            )
        for key, value in schema.items():
            assert_backend_safe(value, path=f"{path}.{key}")
    elif isinstance(schema, list):
        for i, value in enumerate(schema):
            assert_backend_safe(value, path=f"{path}[{i}]")


def build_mention_schema() -> dict[str, Any]:
    """Generate the backend-safe mention JSON Schema from the Pydantic model.

    Returns:
        A schema document ready to hand to a Bedrock tool ``inputSchema`` or an
        OpenAI-compatible ``response_format.json_schema.schema``.
    """
    raw = MentionList.model_json_schema(mode="serialization")
    defs = raw.pop("$defs", {})
    schema = _inline_refs(raw, defs)
    schema = _collapse_nullable(schema)
    schema = _strip_noise(schema)

    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = "MentionList"
    schema["additionalProperties"] = False
    schema.setdefault("type", "object")
    schema["required"] = ["mentions"]

    items = schema.get("properties", {}).get("mentions", {}).get("items")
    if isinstance(items, dict):
        items["additionalProperties"] = False
        # SPEC §6.1: exactly these four are required of a mention.
        items["required"] = ["mention_type", "text", "evidence", "assertion"]

    assert_backend_safe(schema)
    return schema


def write_mention_schema(path: Path | str | None = None) -> Path:
    """Write the generated schema to disk, formatted stably."""
    target = Path(path) if path else SCHEMA_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(build_mention_schema(), indent=2, sort_keys=True) + "\n")
    return target


def load_mention_schema() -> dict[str, Any]:
    """Load the checked-in schema, falling back to generating it.

    The file is the artifact users and backends see; generating on the fly is
    the fallback for an editable checkout where it has not been written yet.
    """
    if SCHEMA_PATH.exists():
        return json.loads(SCHEMA_PATH.read_text())  # type: ignore[no-any-return]
    return build_mention_schema()


def schema_digest(schema: dict[str, Any] | None = None) -> str:
    """Short SHA-256 of the canonical schema, recorded in run summaries.

    Lets a reader tell whether two runs targeted the same decoding schema.
    """
    import hashlib

    blob = json.dumps(schema or build_mention_schema(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
