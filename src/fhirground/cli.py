"""The ``fhirground`` command line.

Exit codes are SPEC §5.1: ``0`` ok, ``1`` some notes failed, ``2`` usage error,
``4`` backend or auth error. The distinction matters in a pipeline -- a
throttled Bedrock call and a note that produced no resources need different
responses, and both are different from a bad flag.

Error messages are escaped before printing: fhirground's input errors end with
a ``[path:line]`` location, and rich would otherwise parse the brackets as
markup and discard the line.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from fhirground.exceptions import (
    BackendError,
    ConfigError,
    FhirgroundError,
    InputError,
)

app = typer.Typer(
    name="fhirground",
    help="Grounded, validated FHIR R4 extraction from clinical notes.",
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()
err_console = Console(stderr=True)

EXIT_OK = 0
EXIT_SOME_FAILED = 1
EXIT_USAGE = 2
EXIT_BACKEND = 4


def _fail(exc: Exception, code: int = EXIT_USAGE) -> None:
    """Print a clean, escaped error and exit with ``code``."""
    err_console.print(f"[bold red]error:[/bold red] {escape(str(exc))}")
    raise typer.Exit(code)


def _build_resolvers(value_sets: Path | None):
    """Build a ResolverConfig from a directory of value-set CSVs.

    Looks for ``conditions``, ``medications``, and ``observations`` by stem.
    A missing file means that mention type stays text-only, which is a
    legitimate configuration rather than an error -- most sites are licensed
    for some terminologies and not others.
    """
    from fhirground.terminology import Chain, LocalValueSet, ResolverConfig

    if value_sets is None:
        return ResolverConfig()
    if not value_sets.exists():
        raise ConfigError(f"value set path not found: {value_sets}")

    def load(*stems: str):
        for stem in stems:
            for suffix in (".csv", ".json"):
                path = value_sets / f"{stem}{suffix}" if value_sets.is_dir() else value_sets
                if path.exists():
                    loader = LocalValueSet.from_csv if suffix == ".csv" else LocalValueSet.from_json
                    return Chain([loader(path)])
        return None

    return ResolverConfig(
        condition=load("conditions", "condition"),
        medication=load("medications", "medication", "meds"),
        observation=load("observations", "observation", "obs"),
    )


def _build_backend(
    backend: str,
    *,
    model: str | None,
    base_url: str | None,
    region: str,
    temperature: float,
    fixtures: Path | None,
):
    """Construct the requested backend, validating its required options."""
    from fhirground.backends import BedrockBackend, MockBackend, OpenAICompatible

    if backend == "mock":
        if fixtures is not None:
            return MockBackend.from_fixture_file(fixtures)
        return MockBackend()
    if backend == "openai_compatible":
        if not base_url:
            raise ConfigError(
                "--backend openai_compatible needs --base-url, e.g. http://localhost:8000/v1"
            )
        if not model:
            raise ConfigError("--backend openai_compatible needs --model")
        return OpenAICompatible(base_url=base_url, model=model, temperature=temperature)
    if backend == "bedrock":
        if not model:
            raise ConfigError(
                "--backend bedrock needs --model (a Bedrock model id or inference profile ARN)"
            )
        return BedrockBackend(model_id=model, region=region, temperature=temperature)
    raise ConfigError(f"unknown backend {backend!r}; use 'mock', 'openai_compatible', or 'bedrock'")


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------


@app.command()
def extract(
    input: Annotated[Path, typer.Option("--input", help="Notes JSONL, or a FHIR directory.")],
    out: Annotated[Path, typer.Option("--out", help="Output directory.")],
    backend: Annotated[
        str, typer.Option("--backend", help="mock, openai_compatible, or bedrock.")
    ] = "mock",
    input_format: Annotated[str, typer.Option("--input-format", help="jsonl or fhir.")] = "jsonl",
    model: Annotated[str | None, typer.Option("--model", help="Model id for the backend.")] = None,
    base_url: Annotated[
        str | None, typer.Option("--base-url", help="OpenAI-compatible server base URL.")
    ] = None,
    region: Annotated[str, typer.Option("--region", help="AWS region for Bedrock.")] = "us-east-1",
    value_sets: Annotated[
        Path | None, typer.Option("--value-sets", help="Directory of value-set CSVs.")
    ] = None,
    fixtures: Annotated[
        Path | None, typer.Option("--fixtures", help="Mock backend fixture JSON.")
    ] = None,
    formats: Annotated[
        str, typer.Option("--format", help="Comma-separated: bundle,ndjson,clinval,debug,summary.")
    ] = "bundle,ndjson,clinval,debug,summary",
    samples: Annotated[int, typer.Option("--samples", help="Self-consistency samples.")] = 1,
    temperature: Annotated[
        float, typer.Option("--temperature", help="Sampling temperature.")
    ] = 0.0,
    max_repairs: Annotated[
        int, typer.Option("--max-repairs", help="Repair attempts after a schema failure.")
    ] = 2,
    min_confidence: Annotated[
        float, typer.Option("--min-confidence", help="Drop resources below this score.")
    ] = 0.0,
    emit_refuted: Annotated[
        bool, typer.Option("--emit-refuted", help="Emit negated conditions as refuted.")
    ] = False,
    no_quotes: Annotated[
        bool, typer.Option("--no-quotes", help="Omit evidence quotes (offsets are kept).")
    ] = False,
    concurrency: Annotated[int, typer.Option("--concurrency", help="Worker threads.")] = 4,
    cache: Annotated[
        Path | None, typer.Option("--cache", help="LLM call cache path. Contains note text.")
    ] = None,
    calibrator: Annotated[
        Path | None, typer.Option("--calibrator", help="Fitted calibrator JSON.")
    ] = None,
    extension_base: Annotated[
        str | None, typer.Option("--extension-base", help="Base URL for fhirground's extensions.")
    ] = None,
) -> None:
    """Extract FHIR resources from clinical notes."""
    from fhirground.backends.prompts import PROMPT_VERSION
    from fhirground.confidence import Calibrator
    from fhirground.export import write_outputs
    from fhirground.io import read_notes
    from fhirground.pipeline import Pipeline, PipelineConfig

    started = datetime.now(UTC)
    try:
        notes = read_notes(input, input_format)
        resolvers = _build_resolvers(value_sets)
        engine = _build_backend(
            backend,
            model=model,
            base_url=base_url,
            region=region,
            temperature=temperature,
            fixtures=fixtures,
        )
        config = PipelineConfig(
            samples=samples,
            temperature=temperature,
            max_repairs=max_repairs,
            min_confidence=min_confidence,
            emit_refuted=emit_refuted,
            include_quotes=not no_quotes,
            cache_path=cache,
            extension_base_url=extension_base,
        )
        cal = Calibrator.load(calibrator) if calibrator else None
        pipe = Pipeline(config, backend=engine, resolvers=resolvers, calibrator=cal)
    except (InputError, ConfigError, FhirgroundError) as exc:
        _fail(exc)
        return

    console.print(
        f"Extracting from [cyan]{len(notes)}[/cyan] note(s) with backend [cyan]{backend}[/cyan]..."
    )

    try:
        results = list(pipe.run(notes, concurrency=concurrency))
    except BackendError as exc:
        _fail(exc, EXIT_BACKEND)
        return
    except FhirgroundError as exc:
        _fail(exc)
        return

    written = write_outputs(
        results,
        out,
        formats=[f.strip() for f in formats.split(",")],
        system_name=f"fhirground-{backend}",
        model_id=getattr(engine, "model_id", backend),
        prompt_version=PROMPT_VERSION,
        started=started,
        backend_description=engine.describe(),
        resolver_description=resolvers.describe(),
        cache_stats=pipe.cache.stats() if pipe.cache else None,
        urls=config.extension_urls,
    )

    _print_run_table(results)
    console.print(f"\nwritten to [cyan]{out}[/cyan]")
    for name, path in sorted(written.items()):
        if isinstance(path, Path):
            console.print(f"  {name:10} {path}")

    failed = [r for r in results if r.errors]
    if failed:
        err_console.print(
            f"[yellow]{len(failed)} of {len(results)} note(s) had chunk-level failures; "
            "see run_summary.json[/yellow]"
        )
        raise typer.Exit(EXIT_SOME_FAILED)


def _print_run_table(results: list[Any]) -> None:
    """Print per-type resource counts and drop reasons."""
    from fhirground.assertion import DropReason

    counts: dict[str, int] = {}
    drops: dict[str, int] = {r.value: 0 for r in DropReason}
    for result in results:
        for key, value in result.resource_counts().items():
            counts[key] = counts.get(key, 0) + value
        for key, value in result.drop_counts().items():
            drops[key] = drops.get(key, 0) + value

    table = Table(title="resources emitted")
    table.add_column("resource type")
    table.add_column("n", justify="right")
    for key in sorted(counts):
        table.add_row(key, str(counts[key]))
    console.print(table)

    nonzero = {k: v for k, v in drops.items() if v}
    if nonzero:
        dropped = Table(title="mentions dropped")
        dropped.add_column("reason")
        dropped.add_column("n", justify="right")
        for key, value in sorted(nonzero.items(), key=lambda kv: -kv[1]):
            dropped.add_row(key, str(value))
        console.print(dropped)

    n_errors = sum(len(r.validation.errors) for r in results)
    n_warnings = sum(len(r.validation.warnings) for r in results)
    if n_errors or n_warnings:
        console.print(f"[yellow]validation: {n_errors} error(s), {n_warnings} warning(s)[/yellow]")
    else:
        console.print("[green]validation: every emitted resource passed[/green]")


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


@app.command()
def validate(
    path: Annotated[Path, typer.Argument(help="Bundle file or directory of them.")],
    hl7_validator: Annotated[
        Path | None, typer.Option("--hl7-validator", help="Path to validator_cli.jar.")
    ] = None,
    notes: Annotated[
        Path | None, typer.Option("--notes", help="Notes JSONL, to check evidence offsets.")
    ] = None,
    extension_base: Annotated[
        str | None, typer.Option("--extension-base", help="Base URL for fhirground's extensions.")
    ] = None,
) -> None:
    """Validate existing output: model parse, project rules, optional HL7 jar."""
    from fhirground.evaluate import read_resources
    from fhirground.io import read_notes_jsonl
    from fhirground.mapping import ExtensionUrls
    from fhirground.validate import ValidationReport, run_hl7_validator, validate_resource

    try:
        resources = read_resources(path)
        note_text = {}
        if notes:
            note_text = {n.note_id: n.text for n in read_notes_jsonl(notes)}
    except (FhirgroundError, FileNotFoundError, json.JSONDecodeError) as exc:
        _fail(exc)
        return

    if not resources:
        _fail(InputError("no resources found", path=str(path)))
        return

    urls = ExtensionUrls(extension_base) if extension_base else ExtensionUrls()
    report = ValidationReport()
    for resource in resources:
        # Offsets can only be checked when the note is available; without it the
        # other rules still run.
        text = None
        for note_id, body in note_text.items():
            if str(resource.get("id", "")).startswith(note_id):
                text = body
                break
        report.extend(validate_resource(resource, text, urls=urls))

    if hl7_validator:
        console.print("Running the HL7 FHIR Validator (this is slow)...")
        report.extend(run_hl7_validator(resources, hl7_validator))

    console.print(f"Checked [cyan]{len(resources)}[/cyan] resource(s)")
    if report.ok and not report.warnings:
        console.print("[green]all checks passed[/green]")
        return

    for finding in report.findings[:50]:
        colour = "red" if finding.severity.value == "error" else "yellow"
        console.print(f"[{colour}]{escape(str(finding))}[/{colour}]")
    if len(report.findings) > 50:
        console.print(f"[dim]... and {len(report.findings) - 50} more[/dim]")

    console.print(f"\n{len(report.errors)} error(s), {len(report.warnings)} warning(s)")
    if report.errors:
        raise typer.Exit(EXIT_SOME_FAILED)


# ---------------------------------------------------------------------------
# synth / score / calibrate / schema
# ---------------------------------------------------------------------------


@app.command()
def synth(
    out: Annotated[Path, typer.Option("--out", help="Output directory.")],
    n_notes: Annotated[int, typer.Option("--n-notes", help="How many notes to render.")] = 50,
    seed: Annotated[int, typer.Option("--seed", help="Seed; output is deterministic.")] = 0,
    synthea_dir: Annotated[
        Path | None,
        typer.Option("--synthea-dir", help="Synthea FHIR output to derive records from."),
    ] = None,
    no_distractors: Annotated[
        bool, typer.Option("--no-distractors", help="Omit negated and family-history distractors.")
    ] = False,
) -> None:
    """Generate synthetic notes with exact gold FHIR.

    With no --synthea-dir, records are generated internally. Synthea is a Java
    program and is not required; scripts/regenerate_synthea.sh documents the
    version and arguments for anyone who has a JVM.
    """
    from fhirground.synth import write_benchmark

    try:
        written = write_benchmark(
            out,
            n_notes=n_notes,
            seed=seed,
            synthea_dir=synthea_dir,
            inject_distractors=not no_distractors,
        )
    except (FhirgroundError, ValueError) as exc:
        _fail(exc)
        return

    manifest = json.loads((out / "manifest.json").read_text())
    console.print(
        f"Rendered [cyan]{manifest['n_notes']}[/cyan] note(s) with "
        f"[cyan]{manifest['n_gold_resources']}[/cyan] gold resource(s) "
        f"from [cyan]{manifest['source']}[/cyan] records"
    )
    table = Table(title="artifacts")
    table.add_column("name")
    table.add_column("path")
    for name, path in sorted(written.items()):
        table.add_row(name, str(path))
    console.print(table)
    console.print("[dim]Every artifact is synthetic. No real patient record was involved.[/dim]")


@app.command()
def score(
    pred: Annotated[
        Path, typer.Option("--pred", help="Predicted NDJSON, bundles, or a directory.")
    ],
    gold: Annotated[Path, typer.Option("--gold", help="Gold NDJSON or directory.")],
    out: Annotated[Path | None, typer.Option("--out", help="Write score.json here.")] = None,
    no_require_code: Annotated[
        bool, typer.Option("--no-require-code", help="Accept a text match where gold is coded.")
    ] = False,
    no_require_status: Annotated[
        bool, typer.Option("--no-require-status", help="Ignore status fields when matching.")
    ] = False,
    tolerance: Annotated[
        float, typer.Option("--tolerance", help="Relative numeric tolerance.")
    ] = 0.01,
    summary: Annotated[
        Path | None, typer.Option("--summary", help="run_summary.json, to include drop reasons.")
    ] = None,
) -> None:
    """Score predicted resources against gold FHIR."""
    from fhirground.evaluate import MatchSpec, read_resources, score_resources

    try:
        predicted = read_resources(pred)
        gold_resources = read_resources(gold)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        _fail(exc)
        return

    drop_reasons: dict[str, int] = {}
    if summary and summary.exists():
        drop_reasons = json.loads(summary.read_text()).get("drop_counts") or {}

    report = score_resources(
        predicted,
        gold_resources,
        MatchSpec(
            require_code=not no_require_code,
            require_status=not no_require_status,
            numeric_tolerance=tolerance,
        ),
        drop_reasons=drop_reasons,
    )

    console.print(report.to_markdown())
    console.print(f"\n[dim]{report.as_dict()['caveat']}[/dim]")
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.as_dict(), indent=2))
        console.print(f"\nwritten to [cyan]{out}[/cyan]")


@app.command()
def calibrate(
    debug: Annotated[Path, typer.Option("--debug", help="debug.jsonl from a run.")],
    gold: Annotated[Path, typer.Option("--gold", help="Gold resources, to label each prediction.")],
    out: Annotated[Path, typer.Option("--out", help="Where to write calibrator.json.")],
    method: Annotated[str, typer.Option("--method", help="isotonic or platt.")] = "isotonic",
) -> None:
    """Fit a confidence calibrator on labelled dev data.

    Labels come from matching each emitted resource against gold, so this needs
    a run whose notes have gold -- the synthetic benchmark, or your own
    annotated set.
    """
    from fhirground.confidence import Calibrator
    from fhirground.evaluate import read_resources, resources_match

    try:
        gold_resources = read_resources(gold)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        _fail(exc)
        return

    if not debug.exists():
        _fail(InputError("debug file not found", path=str(debug)))
        return

    scores: list[float] = []
    labels: list[int] = []
    for line in debug.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        score_value = row.get("score")
        if score_value is None or not row.get("resource_id"):
            continue
        resolution = row.get("resolution") or {}
        probe = {
            "resourceType": row.get("resource_type"),
            "code": {
                "coding": (
                    [{"system": resolution.get("system"), "code": resolution.get("code")}]
                    if resolution.get("code")
                    else []
                ),
                "text": (row.get("mention") or {}).get("text", ""),
            },
        }
        probe["medicationCodeableConcept"] = probe["code"]
        matched = any(resources_match(probe, g, None) for g in gold_resources)
        scores.append(float(score_value))
        labels.append(1 if matched else 0)

    try:
        cal = Calibrator().fit(scores, labels, method=method)
        path = cal.save(out)
    except FhirgroundError as exc:
        _fail(exc)
        return

    console.print(
        f"Fitted a [cyan]{cal.method}[/cyan] calibrator on [cyan]{cal.n_samples}[/cyan] "
        f"labelled example(s) ({sum(labels)} positive)"
    )
    console.print(f"written to [cyan]{path}[/cyan]")
    console.print(
        "[dim]Resources scored with this calibrator record calibrated: true. "
        "Without one, scores are heuristics and not probabilities.[/dim]"
    )


@app.command()
def schema(
    print_: Annotated[bool, typer.Option("--print", help="Print the mention schema.")] = False,
    out: Annotated[Path | None, typer.Option("--out", help="Write the schema here.")] = None,
    check: Annotated[
        bool, typer.Option("--check", help="Verify the checked-in file matches the model.")
    ] = False,
) -> None:
    """Inspect the mention JSON Schema used for constrained decoding."""
    from fhirground.schemas import SCHEMA_PATH, build_mention_schema, schema_digest

    built = build_mention_schema()

    if check:
        if not SCHEMA_PATH.exists():
            _fail(
                ConfigError(
                    f"{SCHEMA_PATH} does not exist; run `fhirground schema --out` to write it"
                )
            )
            return
        on_disk = json.loads(SCHEMA_PATH.read_text())
        if on_disk != built:
            err_console.print(
                "[bold red]error:[/bold red] the checked-in mention.schema.json does not "
                "match the Pydantic model. Regenerate it with `fhirground schema --out`."
            )
            raise typer.Exit(EXIT_SOME_FAILED)
        console.print("[green]the checked-in schema matches the model[/green]")
        return

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(built, indent=2, sort_keys=True) + "\n")
        console.print(f"written to [cyan]{out}[/cyan]")

    if print_ or not out:
        console.print_json(json.dumps(built))
        console.print(
            f"\n[dim]digest {schema_digest(built)} · "
            f"{len(built['properties']['mentions']['items']['properties'])} mention fields, "
            "none of them a code field[/dim]"
        )


@app.command()
def version() -> None:
    """Print versions of fhirground and the FHIR library it pins."""
    from fhirground import __version__
    from fhirground.pipeline import _fhir_library

    console.print(f"fhirground {__version__}")
    console.print(_fhir_library())


def main() -> None:
    """Entry point for ``python -m fhirground``."""
    app()


if __name__ == "__main__":
    main()
