#!/usr/bin/env python
"""fhirground demo: notes to validated FHIR, on the toy data in this repository.

Run it with::

    uv sync
    uv run python demo.py

Everything it reads lives in ``examples/data/`` and was produced by
``fhirground.synth``. The gold FHIR was generated from the same source records
the notes were rendered from, so the benchmark is exact by construction. No
real patient record was involved, and nothing here reaches the network: the
demo uses the deterministic ``mock`` backend.

The demo walks the pipeline and then does three things worth watching:

* it shows the **assertion policy** dropping negated and family-history
  mentions rather than emitting them as facts;
* it tries to smuggle a **hallucinated code** through the mention schema and
  shows the schema rejecting it;
* it scores the output against gold and reports precision *alongside* what was
  dropped, because precision rises when a pipeline emits less.

Output lands in ``demo_output/``.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import fhirground as fg
from fhirground.backends import MockBackend
from fhirground.evaluate import MatchSpec, read_resources, score_resources
from fhirground.export import write_outputs
from fhirground.terminology import Chain, LocalValueSet, ResolverConfig

DATA = Path(__file__).parent / "examples" / "data"
OUT = Path(__file__).parent / "demo_output"

GREEN, YELLOW, RED, DIM, BOLD, RESET = (
    "\033[32m", "\033[33m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"
)
if not sys.stdout.isatty():
    GREEN = YELLOW = RED = DIM = BOLD = RESET = ""


def step(n: int, title: str) -> None:
    print(f"\n{BOLD}[{n}/6] {title}{RESET}")


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def note(msg: str) -> None:
    print(f"  {DIM}{msg}{RESET}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}!{RESET} {msg}")


def main() -> int:
    print(f"{BOLD}fhirground {fg.__version__} — demo{RESET}")
    print(f"{DIM}Synthetic data only. No PHI, no network calls, mock backend.{RESET}")

    if not DATA.exists():
        print(f"{RED}error:{RESET} {DATA} is missing. Regenerate it with:")
        print("  uv run fhirground synth --out examples/data --n-notes 25 --seed 0")
        return 1

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    # ------------------------------------------------------------------
    step(1, "Load the toy notes, value sets, and gold")
    # ------------------------------------------------------------------
    notes = fg.read_notes_jsonl(DATA / "notes.jsonl")
    manifest = json.loads((DATA / "manifest.json").read_text())
    resolvers = ResolverConfig(
        condition=Chain([LocalValueSet.from_csv(DATA / "valuesets" / "conditions.csv")]),
        medication=Chain([LocalValueSet.from_csv(DATA / "valuesets" / "medications.csv")]),
        observation=Chain([LocalValueSet.from_csv(DATA / "valuesets" / "observations.csv")]),
    )
    gold = read_resources(DATA / "gold")

    ok(f"{len(notes)} synthetic notes, {len(gold)} gold resources")
    ok(f"value sets cover {len(resolvers.all_known_codes())} codes across 3 terminologies")
    note(f"source: {manifest['source']} records, seed {manifest['seed']}")
    note(
        f"{len(manifest['distractors'])} distractors injected "
        "(negated, family-history, hedged) that must not be emitted as facts"
    )

    # ------------------------------------------------------------------
    step(2, "Show what the LLM is allowed to say")
    # ------------------------------------------------------------------
    schema = fg.build_mention_schema()
    fields = list(schema["properties"]["mentions"]["items"]["properties"])
    ok(f"the mention schema has {len(fields)} fields and {len(schema['properties'])} top-level key")
    ok("none of them is a code field — a model has nowhere to put a SNOMED or LOINC code")
    note(f"required: {schema['properties']['mentions']['items']['required']}")

    import jsonschema

    validator = jsonschema.Draft202012Validator(schema)
    smuggled = {
        "mentions": [
            {
                "mention_type": "condition",
                "text": "type 2 diabetes",
                "evidence": "Type 2 diabetes",
                "assertion": "present",
                "snomed_code": "44054006",
            }
        ]
    }
    errors = [e.message for e in validator.iter_errors(smuggled)]
    if errors:
        ok("a model trying to add `snomed_code` is rejected by the schema itself")
        note(f"{errors[0]}")
    else:
        print(f"  {RED}✗{RESET} the schema accepted a code field — this is a bug")
        return 1

    # ------------------------------------------------------------------
    step(3, "Run the pipeline")
    # ------------------------------------------------------------------
    backend = MockBackend()
    pipe = fg.Pipeline(fg.PipelineConfig(), backend=backend, resolvers=resolvers)
    results = list(pipe.run(notes, concurrency=4))

    counts: dict[str, int] = {}
    for result in results:
        for key, value in result.resource_counts().items():
            counts[key] = counts.get(key, 0) + value

    print()
    print(f"  {'resource type':<22} {'n':>6}")
    print(f"  {'-' * 22} {'-' * 6}")
    for key in sorted(counts):
        print(f"  {key:<22} {counts[key]:>6}")

    n_validation_errors = sum(len(r.validation.errors) for r in results)
    if n_validation_errors == 0:
        ok("every emitted resource parses as FHIR R4B and passes the project rules")
    else:
        warn(f"{n_validation_errors} validation error(s)")

    failed = [r for r in results if r.errors]
    if failed:
        warn(f"{len(failed)} note(s) had chunk-level failures")
    else:
        ok(f"all {len(results)} notes processed without a chunk failure")

    # ------------------------------------------------------------------
    step(4, "Show what was dropped, and why")
    # ------------------------------------------------------------------
    drops: dict[str, int] = {}
    for result in results:
        for key, value in result.drop_counts().items():
            drops[key] = drops.get(key, 0) + value

    total_drops = sum(drops.values())
    print()
    for reason, count in sorted(drops.items(), key=lambda kv: -kv[1]):
        bar = "█" * min(40, round(count / max(total_drops, 1) * 40))
        label = "none" if count == 0 else str(count)
        print(f"    {reason:<22} {label:>6}  {bar}")
    note(
        f"{total_drops} mentions declined. Negated and family-history findings are "
        "never emitted as affirmed facts."
    )

    example = next(
        (
            (r.note, m, reason)
            for r in results
            for m, reason in r.dropped
            if reason is fg.DropReason.NEGATED
        ),
        None,
    )
    if example:
        source, mention, reason = example
        print()
        print(f"  {BOLD}One declined mention, in full{RESET}")
        print(f"    note      {source.note_id}")
        print(f"    finding   {mention.text!r} ({mention.mention_type.value})")
        print(f"    assertion {mention.assertion.value}")
        print(f"    evidence  {mention.evidence[:80]!r}")
        print(f"    outcome   dropped as {reason.value}")

    # ------------------------------------------------------------------
    step(5, "Check every claim is grounded in the note")
    # ------------------------------------------------------------------
    checked = 0
    strategies: dict[str, int] = {}
    for result in results:
        for resource in result.resources:
            if resource["resourceType"] == "Provenance":
                continue
            for span in fg.mapping.read_evidence(resource):
                actual = result.note.text[span["start"] : span["end"]]
                if actual != span["quote"]:
                    print(f"  {RED}✗{RESET} offset/quote mismatch in {resource['id']}")
                    return 1
                checked += 1
                key = span.get("grounding") or "?"
                strategies[key] = strategies.get(key, 0) + 1

    ok(f"{checked} evidence spans verified: note_text[start:end] equals the stored quote")
    note("grounding strategies: " + ", ".join(f"{k}={v}" for k, v in sorted(strategies.items())))

    allowed = resolvers.all_known_codes()
    emitted_codes: set[tuple[str, str]] = set()
    for result in results:
        for resource in result.resources:
            concept = resource.get("code") or resource.get("medicationCodeableConcept") or {}
            for coding in concept.get("coding") or []:
                emitted_codes.add((coding["system"], coding["code"]))
    invented = emitted_codes - allowed
    if invented:
        print(f"  {RED}✗{RESET} codes not in any value set: {sorted(invented)[:3]}")
        return 1
    ok(f"all {len(emitted_codes)} distinct emitted codes came from a terminology resolver")

    # ------------------------------------------------------------------
    step(6, "Write outputs and score against gold")
    # ------------------------------------------------------------------
    written = write_outputs(
        results,
        OUT,
        system_name="fhirground-mock",
        model_id=backend.model_id,
        prompt_version=fg.backends.PROMPT_VERSION,
        backend_description=backend.describe(),
        resolver_description=resolvers.describe(),
    )

    predicted = [
        r for result in results for r in result.resources if r["resourceType"] != "Provenance"
    ]
    report = score_resources(
        predicted, gold, MatchSpec(), drop_reasons=drops
    )
    (OUT / "score.json").write_text(json.dumps(report.as_dict(), indent=2))

    print()
    for line in report.to_markdown().splitlines():
        print(f"  {line}")
    print()
    warn(
        "Precision rises when a pipeline emits less, so read it next to recall and the "
        "drop table above."
    )
    note(
        "These numbers measure recovery of the records a synthetic note was rendered "
        "from. They are not a clinical accuracy claim."
    )

    print()
    print(f"{BOLD}Done.{RESET} Artifacts in {OUT.name}/:")
    for name, path in sorted(written.items()):
        if isinstance(path, Path):
            print(f"  {name:<10} {path.relative_to(Path.cwd())}")
        elif isinstance(path, dict):
            print(f"  {name:<10} {len(path)} files in {OUT.name}/ndjson/")
        elif isinstance(path, list):
            print(f"  {name:<10} {len(path)} files in {OUT.name}/bundles/")
    print(f"  {'score':<10} {(OUT / 'score.json').relative_to(Path.cwd())}")

    sample = sorted((OUT / "bundles").glob("*.json"))[:1]
    if sample:
        print(f"\nLook at {BOLD}{sample[0].relative_to(Path.cwd())}{RESET} for a full Bundle,")
        print(f"or {BOLD}{(OUT / 'debug.jsonl').relative_to(Path.cwd())}{RESET} "
              "for one trace per mention.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
