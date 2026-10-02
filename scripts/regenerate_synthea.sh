#!/usr/bin/env bash
# Regenerate the Synthea source records fhirground's benchmark can use.
#
# fhirground does NOT require Synthea. `fhirground synth` generates equivalent
# records internally, which is what the committed set in examples/data/ was
# built with -- the development and CI environments have no JVM. See
# docs/decisions.md D-006.
#
# This script exists so that anyone who *does* have Java can reproduce a
# Synthea-derived benchmark with a pinned version and recorded arguments.
#
# Requires: Java 11+ and network access on first run.

set -euo pipefail

SYNTHEA_VERSION="3.3.0"
SEED="${SEED:-0}"
POPULATION="${POPULATION:-50}"
STATE="${STATE:-Massachusetts}"
OUTDIR="${OUTDIR:-tests/fixtures/synthea}"

if ! command -v java >/dev/null 2>&1; then
  echo "error: no 'java' on PATH. Synthea is a Java program." >&2
  echo "       Run 'fhirground synth --out examples/data' instead, which needs no JVM." >&2
  exit 1
fi

JAR="synthea-with-dependencies-${SYNTHEA_VERSION}.jar"
URL="https://github.com/synthetichealth/synthea/releases/download/v${SYNTHEA_VERSION}/synthea-with-dependencies.jar"

if [ ! -f "$JAR" ]; then
  echo "Downloading Synthea ${SYNTHEA_VERSION}..."
  curl -fsSL "$URL" -o "$JAR"
fi

mkdir -p "$OUTDIR"

# -p population, -s seed, -cs clinician seed (pinned for reproducibility)
# exporter.fhir.export=true          -> FHIR R4 bundles
# exporter.fhir.use_r4=true          -> R4 rather than R5
# exporter.clinical_note.export=true -> DocumentReference notes, used by
#                                       --input-format fhir
java -jar "$JAR" \
  -p "$POPULATION" \
  -s "$SEED" \
  -cs "$SEED" \
  --exporter.baseDirectory "$OUTDIR" \
  --exporter.fhir.export true \
  --exporter.fhir.use_r4 true \
  --exporter.clinical_note.export true \
  --exporter.hospital.fhir.export false \
  --exporter.practitioner.fhir.export false \
  "$STATE"

echo
echo "Synthea ${SYNTHEA_VERSION} output written to ${OUTDIR}/fhir/"
echo "Build a benchmark from it with:"
echo "  fhirground synth --synthea-dir ${OUTDIR}/fhir --n-notes ${POPULATION} --seed ${SEED} --out examples/data"
