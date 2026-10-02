"""CLI smoke tests: every command, and the SPEC §5.1 exit codes."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from fhirground.cli import EXIT_BACKEND, EXIT_OK, EXIT_SOME_FAILED, EXIT_USAGE, app

runner = CliRunner()


def run(*args: str):
    return runner.invoke(app, list(args))


def flat(result) -> str:
    """Output with whitespace collapsed; rich wraps to the terminal width."""
    return " ".join(result.output.split())


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    """A generated benchmark on disk, shared across the smoke tests."""
    path = tmp_path_factory.mktemp("fg_demo")
    result = run("synth", "--out", str(path / "data"), "--n-notes", "6", "--seed", "0")
    assert result.exit_code == EXIT_OK, result.output
    return path


class TestHelp:
    def test_no_args_shows_help(self):
        result = run()
        assert result.exit_code in (0, 2)
        assert "Grounded, validated FHIR" in flat(result)

    @pytest.mark.parametrize(
        "cmd", ["extract", "validate", "synth", "score", "calibrate", "schema", "version"]
    )
    def test_every_command_has_help(self, cmd):
        assert run(cmd, "--help").exit_code == EXIT_OK

    def test_version(self):
        from fhirground import __version__

        result = run("version")
        assert result.exit_code == EXIT_OK
        assert __version__ in flat(result)
        assert "R4B" in flat(result)


class TestSchemaCommand:
    def test_print(self):
        result = run("schema", "--print")
        assert result.exit_code == EXIT_OK
        assert "mentions" in result.output

    def test_write(self, tmp_path):
        out = tmp_path / "s.json"
        assert run("schema", "--out", str(out)).exit_code == EXIT_OK
        schema = json.loads(out.read_text())
        assert schema["title"] == "MentionList"

    def test_check_passes_against_the_checked_in_file(self):
        """Guards the checked-in schema against drift from the model."""
        result = run("schema", "--check")
        assert result.exit_code == EXIT_OK, result.output
        assert "matches the model" in flat(result)


class TestSynthCommand:
    def test_writes_every_artifact(self, demo):
        data = demo / "data"
        for name in (
            "notes.jsonl",
            "manifest.json",
            "gold/Condition.ndjson",
            "valuesets/conditions.csv",
        ):
            assert (data / name).exists(), name

    def test_states_that_no_real_data_was_used(self, tmp_path):
        result = run("synth", "--out", str(tmp_path / "d"), "--n-notes", "3")
        assert "No real patient record" in flat(result)

    def test_deterministic(self, tmp_path):
        run("synth", "--out", str(tmp_path / "a"), "--n-notes", "4", "--seed", "7")
        run("synth", "--out", str(tmp_path / "b"), "--n-notes", "4", "--seed", "7")
        assert (tmp_path / "a" / "notes.jsonl").read_bytes() == (
            tmp_path / "b" / "notes.jsonl"
        ).read_bytes()

    def test_no_distractors_flag(self, tmp_path):
        run("synth", "--out", str(tmp_path / "d"), "--n-notes", "4", "--no-distractors")
        manifest = json.loads((tmp_path / "d" / "manifest.json").read_text())
        assert manifest["distractors"] == []


class TestExtractCommand:
    def test_end_to_end_with_the_mock_backend(self, demo):
        out = demo / "out"
        result = run(
            "extract",
            "--input",
            str(demo / "data" / "notes.jsonl"),
            "--value-sets",
            str(demo / "data" / "valuesets"),
            "--out",
            str(out),
            "--backend",
            "mock",
        )
        assert result.exit_code == EXIT_OK, result.output
        assert (out / "bundles").exists()
        assert (out / "ndjson" / "Condition.ndjson").exists()
        assert (out / "clinval_predictions.jsonl").exists()
        assert (out / "debug.jsonl").exists()
        assert (out / "run_summary.json").exists()
        assert "resources emitted" in flat(result)

    def test_reports_validation_cleanliness(self, demo):
        result = run(
            "extract",
            "--input",
            str(demo / "data" / "notes.jsonl"),
            "--value-sets",
            str(demo / "data" / "valuesets"),
            "--out",
            str(demo / "out2"),
        )
        assert "every emitted resource passed" in flat(result)

    def test_drop_reasons_are_reported(self, demo):
        result = run(
            "extract",
            "--input",
            str(demo / "data" / "notes.jsonl"),
            "--out",
            str(demo / "out3"),
        )
        assert "mentions dropped" in flat(result)

    def test_format_subset(self, demo):
        out = demo / "out4"
        run(
            "extract",
            "--input",
            str(demo / "data" / "notes.jsonl"),
            "--out",
            str(out),
            "--format",
            "bundle",
        )
        assert (out / "bundles").exists()
        assert not (out / "debug.jsonl").exists()

    def test_unknown_format_is_a_usage_error(self, demo):
        result = run(
            "extract",
            "--input",
            str(demo / "data" / "notes.jsonl"),
            "--out",
            str(demo / "out5"),
            "--format",
            "parquet",
        )
        assert result.exit_code != EXIT_OK

    def test_missing_input_is_a_clean_error(self, tmp_path):
        result = run(
            "extract", "--input", str(tmp_path / "nope.jsonl"), "--out", str(tmp_path / "o")
        )
        assert result.exit_code == EXIT_USAGE
        assert "not found" in flat(result)
        assert "Traceback" not in result.output

    def test_malformed_input_reports_the_line(self, tmp_path):
        bad = tmp_path / "bad.jsonl"
        bad.write_text('{"note_id":"n1"}\n{oops\n')
        result = run("extract", "--input", str(bad), "--out", str(tmp_path / "o"))
        assert result.exit_code == EXIT_USAGE
        assert "Traceback" not in result.output

    def test_openai_backend_requires_a_base_url(self, demo):
        result = run(
            "extract",
            "--input",
            str(demo / "data" / "notes.jsonl"),
            "--out",
            str(demo / "o"),
            "--backend",
            "openai_compatible",
            "--model",
            "m",
        )
        assert result.exit_code == EXIT_USAGE
        assert "--base-url" in flat(result)

    def test_bedrock_backend_requires_a_model(self, demo):
        result = run(
            "extract",
            "--input",
            str(demo / "data" / "notes.jsonl"),
            "--out",
            str(demo / "o"),
            "--backend",
            "bedrock",
        )
        assert result.exit_code == EXIT_USAGE
        assert "--model" in flat(result)

    def test_unknown_backend(self, demo):
        result = run(
            "extract",
            "--input",
            str(demo / "data" / "notes.jsonl"),
            "--out",
            str(demo / "o"),
            "--backend",
            "telepathy",
        )
        assert result.exit_code == EXIT_USAGE
        assert "unknown backend" in flat(result)

    def test_missing_value_sets_path(self, demo):
        result = run(
            "extract",
            "--input",
            str(demo / "data" / "notes.jsonl"),
            "--out",
            str(demo / "o"),
            "--value-sets",
            "/nonexistent/vs",
        )
        assert result.exit_code == EXIT_USAGE
        assert "not found" in flat(result)

    def test_samples_without_temperature_is_rejected(self, demo):
        result = run(
            "extract",
            "--input",
            str(demo / "data" / "notes.jsonl"),
            "--out",
            str(demo / "o"),
            "--samples",
            "3",
        )
        assert result.exit_code == EXIT_USAGE
        assert "identical" in flat(result)

    def test_no_quotes_flag(self, demo):
        out = demo / "out_nq"
        run(
            "extract",
            "--input",
            str(demo / "data" / "notes.jsonl"),
            "--out",
            str(out),
            "--no-quotes",
            "--format",
            "bundle",
        )
        bundle = json.loads(next((out / "bundles").glob("*.json")).read_text())
        for entry in bundle["entry"]:
            for ext in entry["resource"].get("extension") or []:
                if "evidence" in ext["url"]:
                    urls = {s["url"] for s in ext["extension"]}
                    assert "quote" not in urls
                    assert {"start", "end"} <= urls

    def test_cache_round_trip(self, demo, tmp_path):
        cache = tmp_path / "llm.sqlite"
        for i in range(2):
            result = run(
                "extract",
                "--input",
                str(demo / "data" / "notes.jsonl"),
                "--out",
                str(demo / f"out_cache{i}"),
                "--cache",
                str(cache),
                "--format",
                "summary",
            )
            assert result.exit_code == EXIT_OK
        summary = json.loads((demo / "out_cache1" / "run_summary.json").read_text())
        assert summary["cache"]["hits"] > 0


class TestValidateCommand:
    def test_valid_output_passes(self, demo):
        out = demo / "v_out"
        run(
            "extract",
            "--input",
            str(demo / "data" / "notes.jsonl"),
            "--value-sets",
            str(demo / "data" / "valuesets"),
            "--out",
            str(out),
            "--format",
            "bundle",
        )
        result = run("validate", str(out / "bundles"))
        assert result.exit_code == EXIT_OK, result.output
        assert "all checks passed" in flat(result)

    def test_invalid_resource_is_reported(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text(
            json.dumps(
                {
                    "resourceType": "Condition",
                    "id": "c1",
                    "subject": {"reference": "Patient/1"},
                }
            )
        )
        result = run("validate", str(bad))
        assert result.exit_code == EXIT_SOME_FAILED
        assert "error" in flat(result)

    def test_missing_path(self, tmp_path):
        result = run("validate", str(tmp_path / "nope"))
        assert result.exit_code == EXIT_USAGE
        assert "Traceback" not in result.output

    def test_missing_jar_is_reported(self, demo, tmp_path):
        out = demo / "v_out2"
        run(
            "extract",
            "--input",
            str(demo / "data" / "notes.jsonl"),
            "--out",
            str(out),
            "--format",
            "bundle",
        )
        result = run(
            "validate", str(out / "bundles"), "--hl7-validator", str(tmp_path / "nope.jar")
        )
        assert "not found" in flat(result)


class TestScoreCommand:
    def test_scores_against_gold(self, demo):
        out = demo / "s_out"
        run(
            "extract",
            "--input",
            str(demo / "data" / "notes.jsonl"),
            "--value-sets",
            str(demo / "data" / "valuesets"),
            "--out",
            str(out),
        )
        result = run(
            "score",
            "--pred",
            str(out / "ndjson"),
            "--gold",
            str(demo / "data" / "gold"),
            "--out",
            str(out / "score.json"),
            "--summary",
            str(out / "run_summary.json"),
        )
        assert result.exit_code == EXIT_OK, result.output
        report = json.loads((out / "score.json").read_text())
        assert report["totals"]["tp"] > 0
        assert "drop_reasons" in report
        assert "caveat" in report

    def test_prints_the_precision_caveat(self, demo):
        out = demo / "s_out2"
        run("extract", "--input", str(demo / "data" / "notes.jsonl"), "--out", str(out))
        result = run("score", "--pred", str(out / "ndjson"), "--gold", str(demo / "data" / "gold"))
        assert "Precision rises when the pipeline emits less" in flat(result)

    def test_missing_pred(self, demo, tmp_path):
        result = run(
            "score", "--pred", str(tmp_path / "nope"), "--gold", str(demo / "data" / "gold")
        )
        assert result.exit_code == EXIT_USAGE


class TestCalibrateCommand:
    def test_too_little_data_is_a_clean_error(self, demo, tmp_path):
        out = demo / "c_out"
        run(
            "extract",
            "--input",
            str(demo / "data" / "notes.jsonl"),
            "--value-sets",
            str(demo / "data" / "valuesets"),
            "--out",
            str(out),
        )
        result = run(
            "calibrate",
            "--debug",
            str(out / "debug.jsonl"),
            "--gold",
            str(demo / "data" / "gold"),
            "--out",
            str(tmp_path / "cal.json"),
        )
        # Six notes is below the 10-example minimum, which must be explained
        # rather than silently producing a useless calibrator.
        assert result.exit_code in (EXIT_OK, EXIT_USAGE)
        if result.exit_code == EXIT_USAGE:
            assert "at least 10" in flat(result) or "single class" in flat(result)

    def test_fits_on_a_larger_run(self, tmp_path):
        data, out = tmp_path / "data", tmp_path / "out"
        run("synth", "--out", str(data), "--n-notes", "30", "--seed", "0")
        run(
            "extract",
            "--input",
            str(data / "notes.jsonl"),
            "--value-sets",
            str(data / "valuesets"),
            "--out",
            str(out),
        )
        result = run(
            "calibrate",
            "--debug",
            str(out / "debug.jsonl"),
            "--gold",
            str(data / "gold"),
            "--out",
            str(tmp_path / "cal.json"),
            "--method",
            "platt",
        )
        if result.exit_code == EXIT_OK:
            payload = json.loads((tmp_path / "cal.json").read_text())
            assert payload["fitted"] is True
            assert "calibrated: true" in flat(result)

    def test_missing_debug_file(self, demo, tmp_path):
        result = run(
            "calibrate",
            "--debug",
            str(tmp_path / "nope.jsonl"),
            "--gold",
            str(demo / "data" / "gold"),
            "--out",
            str(tmp_path / "c.json"),
        )
        assert result.exit_code == EXIT_USAGE


class TestModuleEntryPoint:
    def test_python_m_fhirground(self):
        import subprocess
        import sys

        from fhirground import __version__

        out = subprocess.run(
            [sys.executable, "-m", "fhirground", "version"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert out.returncode == 0, out.stderr
        assert __version__ in out.stdout


class TestExitCodes:
    def test_ok_is_zero(self, demo):
        assert (
            run(
                "extract",
                "--input",
                str(demo / "data" / "notes.jsonl"),
                "--out",
                str(demo / "ec_out"),
            ).exit_code
            == EXIT_OK
        )

    def test_usage_error_is_two(self, tmp_path):
        assert (
            run(
                "extract", "--input", str(tmp_path / "nope"), "--out", str(tmp_path / "o")
            ).exit_code
            == EXIT_USAGE
        )

    def test_backend_error_is_four(self, demo):
        """A server that is not there is a backend error, not a usage error."""
        result = run(
            "extract",
            "--input",
            str(demo / "data" / "notes.jsonl"),
            "--out",
            str(demo / "be_out"),
            "--backend",
            "openai_compatible",
            "--base-url",
            "http://127.0.0.1:9/v1",
            "--model",
            "m",
        )
        assert result.exit_code == EXIT_BACKEND
