"""A resource-level scorer against gold FHIR.

For synthetic benchmarks, where gold resources were generated from the same
source records the notes were rendered from. It answers "did we recover the
record" rather than "was the text understood", and those are different
questions -- for the second one, use clinval.

A match needs all of:

1. the same resource type;
2. the same code, or the same normalized text when neither side is coded;
3. compatible status fields -- a ``refuted`` Condition does not match a
   ``confirmed`` one, which is the whole point of assertion handling;
4. numeric values within tolerance (1% relative by default, exact for
   integers).

The report also includes **drop-reason accounting**: how many gold negatives
were correctly dropped. A pipeline that drops everything scores perfect
precision, so recall and the drop breakdown have to be read together.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Status fields that must agree for a match, per resource type.
STATUS_FIELDS: dict[str, tuple[str, ...]] = {
    "Condition": ("verificationStatus", "clinicalStatus"),
    "MedicationRequest": ("status", "intent"),
    "Observation": ("status",),
}


@dataclass(frozen=True, slots=True)
class MatchSpec:
    """How strictly two resources must agree to count as the same.

    Attributes:
        require_code: A coded gold resource must be matched by the same code.
            With this off, a text match is accepted, which measures concept
            recall rather than coding accuracy.
        require_status: Status fields must agree.
        numeric_tolerance: Relative tolerance for a numeric value.
        exact_integers: Integer-valued quantities must match exactly, because
            a heart rate of 72 and 73 are different observations however small
            the relative difference.
        match_text_fallback: When neither side is coded, compare normalized
            text.
    """

    require_code: bool = True
    require_status: bool = True
    numeric_tolerance: float = 0.01
    exact_integers: bool = True
    match_text_fallback: bool = True


@dataclass
class TypeScore:
    """Precision, recall, and F1 for one resource type."""

    resource_type: str
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float | None:
        """TP / (TP + FP), or None when nothing was predicted."""
        denominator = self.tp + self.fp
        return self.tp / denominator if denominator else None

    @property
    def recall(self) -> float | None:
        """TP / (TP + FN), or None when there was no gold."""
        denominator = self.tp + self.fn
        return self.tp / denominator if denominator else None

    @property
    def f1(self) -> float | None:
        """Harmonic mean, or None when either side is undefined."""
        p, r = self.precision, self.recall
        if p is None or r is None or (p + r) == 0:
            return None
        return 2 * p * r / (p + r)

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe row. Undefined metrics are ``null``, never 0."""
        return {
            "resource_type": self.resource_type,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "n_gold": self.tp + self.fn,
            "n_predicted": self.tp + self.fp,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


@dataclass
class ScoreReport:
    """The full scoring result."""

    per_type: dict[str, TypeScore] = field(default_factory=dict)
    unmatched_predicted: list[dict[str, Any]] = field(default_factory=list)
    unmatched_gold: list[dict[str, Any]] = field(default_factory=list)
    drop_reasons: dict[str, int] = field(default_factory=dict)
    spec: MatchSpec = field(default_factory=MatchSpec)

    @property
    def totals(self) -> TypeScore:
        """Micro-averaged counts across types."""
        total = TypeScore(resource_type="ALL")
        for score in self.per_type.values():
            total.tp += score.tp
            total.fp += score.fp
            total.fn += score.fn
        return total

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe report, the shape written to ``score.json``."""
        return {
            "match_spec": {
                "require_code": self.spec.require_code,
                "require_status": self.spec.require_status,
                "numeric_tolerance": self.spec.numeric_tolerance,
                "exact_integers": self.spec.exact_integers,
                "match_text_fallback": self.spec.match_text_fallback,
            },
            "totals": self.totals.as_dict(),
            "per_type": {k: v.as_dict() for k, v in sorted(self.per_type.items())},
            "drop_reasons": self.drop_reasons,
            "n_unmatched_predicted": len(self.unmatched_predicted),
            "n_unmatched_gold": len(self.unmatched_gold),
            "caveat": (
                "Precision rises when the pipeline emits less, so read recall and "
                "drop_reasons alongside it. These numbers measure recovery of the "
                "source records a synthetic note was rendered from, not clinical "
                "accuracy."
            ),
        }

    def to_markdown(self, digits: int = 3) -> str:
        """Render as a Markdown table."""

        def fmt(value: float | None) -> str:
            return "n/a" if value is None else f"{value:.{digits}f}"

        lines = [
            "| resource type | gold | predicted | TP | FP | FN | precision | recall | F1 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for name in sorted(self.per_type):
            s = self.per_type[name]
            lines.append(
                f"| {name} | {s.tp + s.fn} | {s.tp + s.fp} | {s.tp} | {s.fp} | {s.fn} | "
                f"{fmt(s.precision)} | {fmt(s.recall)} | {fmt(s.f1)} |"
            )
        t = self.totals
        lines.append(
            f"| **all** | {t.tp + t.fn} | {t.tp + t.fp} | {t.tp} | {t.fp} | {t.fn} | "
            f"{fmt(t.precision)} | {fmt(t.recall)} | {fmt(t.f1)} |"
        )
        return "\n".join(lines)


def _concept(resource: dict[str, Any]) -> dict[str, Any]:
    """The resource's primary CodeableConcept."""
    return resource.get("code") or resource.get("medicationCodeableConcept") or {}


def _codes(resource: dict[str, Any]) -> set[tuple[str, str]]:
    """Coded identities of a resource."""
    return {
        (str(c.get("system")), str(c.get("code")))
        for c in _concept(resource).get("coding") or []
        if isinstance(c, dict) and c.get("code")
    }


def _text(resource: dict[str, Any]) -> str:
    """Normalized text of the primary concept."""
    return " ".join((_concept(resource).get("text") or "").lower().split())


def _statuses(resource: dict[str, Any]) -> dict[str, frozenset[str]]:
    """Status field values, as sets of codes."""
    resource_type = str(resource.get("resourceType"))
    out: dict[str, frozenset[str]] = {}
    for field_name in STATUS_FIELDS.get(resource_type, ()):
        value = resource.get(field_name)
        if isinstance(value, str):
            out[field_name] = frozenset({value})
        elif isinstance(value, dict):
            out[field_name] = frozenset(
                str(c.get("code"))
                for c in value.get("coding") or []
                if isinstance(c, dict) and c.get("code")
            )
        else:
            out[field_name] = frozenset()
    return out


def _numeric(resource: dict[str, Any]) -> float | None:
    """The resource's numeric value, if it has one."""
    quantity = resource.get("valueQuantity")
    if isinstance(quantity, dict) and quantity.get("value") is not None:
        try:
            return float(quantity["value"])
        except (TypeError, ValueError):
            return None
    return None


def resources_match(
    predicted: dict[str, Any], gold: dict[str, Any], spec: MatchSpec | None = None
) -> bool:
    """Whether two resources denote the same finding under ``spec``."""
    s = spec or MatchSpec()

    if predicted.get("resourceType") != gold.get("resourceType"):
        return False

    pred_codes, gold_codes = _codes(predicted), _codes(gold)
    if gold_codes and pred_codes:
        if not (pred_codes & gold_codes):
            return False
    elif s.require_code and gold_codes and not pred_codes:
        return False
    elif s.match_text_fallback:
        pt, gt = _text(predicted), _text(gold)
        if not pt or not gt or not (pt in gt or gt in pt):
            return False
    else:
        return False

    if s.require_status:
        pred_status, gold_status = _statuses(predicted), _statuses(gold)
        for field_name, gold_values in gold_status.items():
            if not gold_values:
                continue
            pred_values = pred_status.get(field_name, frozenset())
            if pred_values and not (pred_values & gold_values):
                return False

    pred_value, gold_value = _numeric(predicted), _numeric(gold)
    if gold_value is not None and pred_value is not None:
        if s.exact_integers and float(gold_value).is_integer():
            if pred_value != gold_value:
                return False
        else:
            scale = abs(gold_value) or 1.0
            if abs(pred_value - gold_value) / scale > s.numeric_tolerance:
                return False

    return True


def score_resources(
    predicted: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    match: MatchSpec | None = None,
    *,
    drop_reasons: dict[str, int] | None = None,
) -> ScoreReport:
    """Score predicted resources against gold.

    Matching is greedy and one-to-one: each gold resource is claimed by at
    most one prediction, so emitting a finding twice costs a false positive
    rather than earning two true positives.

    Args:
        predicted: Predicted resources. ``Provenance`` is ignored.
        gold: Gold resources.
        match: Matching strictness.
        drop_reasons: Drop counts from the run, carried into the report so
            precision can be read alongside what was declined.

    Returns:
        A :class:`ScoreReport`.
    """
    spec = match or MatchSpec()
    pred = [r for r in predicted if r.get("resourceType") != "Provenance"]
    gold_list = [r for r in gold if r.get("resourceType") != "Provenance"]

    report = ScoreReport(spec=spec, drop_reasons=dict(drop_reasons or {}))
    for resource_type in sorted({str(r.get("resourceType")) for r in pred + gold_list}):
        report.per_type[resource_type] = TypeScore(resource_type=resource_type)

    claimed: set[int] = set()
    for resource in pred:
        resource_type = str(resource.get("resourceType"))
        score = report.per_type[resource_type]
        hit = None
        for i, candidate in enumerate(gold_list):
            if i in claimed:
                continue
            if resources_match(resource, candidate, spec):
                hit = i
                break
        if hit is None:
            score.fp += 1
            report.unmatched_predicted.append(_brief(resource))
        else:
            claimed.add(hit)
            score.tp += 1

    for i, candidate in enumerate(gold_list):
        if i in claimed:
            continue
        resource_type = str(candidate.get("resourceType"))
        report.per_type[resource_type].fn += 1
        report.unmatched_gold.append(_brief(candidate))

    return report


def _brief(resource: dict[str, Any]) -> dict[str, Any]:
    """A short identification of a resource, for the unmatched lists."""
    codes = sorted(_codes(resource))
    return {
        "resourceType": resource.get("resourceType"),
        "id": resource.get("id"),
        "code": codes[0][1] if codes else None,
        "text": _text(resource) or None,
        "value": _numeric(resource),
    }


def read_resources(path: str | Path) -> list[dict[str, Any]]:
    """Read resources from NDJSON, a Bundle, a directory, or a JSON array."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"no such path: {source}")

    files = (
        sorted([*source.glob("**/*.ndjson"), *source.glob("**/*.json")])
        if source.is_dir()
        else [source]
    )
    out: list[dict[str, Any]] = []
    for file in files:
        text = file.read_text(encoding="utf-8").strip()
        if not text:
            continue
        if file.suffix == ".ndjson":
            for line in text.splitlines():
                if line.strip():
                    out.append(json.loads(line))
            continue
        data = json.loads(text)
        out.extend(_flatten(data))
    return out


def _flatten(data: Any) -> Iterable[dict[str, Any]]:
    """Yield resources from a Bundle, a list, or a bare resource."""
    if isinstance(data, list):
        for item in data:
            yield from _flatten(item)
        return
    if not isinstance(data, dict):
        return
    if data.get("resourceType") == "Bundle":
        for entry in data.get("entry") or []:
            resource = entry.get("resource") if isinstance(entry, dict) else None
            if isinstance(resource, dict):
                yield from _flatten(resource)
        return
    if "resourceType" in data:
        yield data
