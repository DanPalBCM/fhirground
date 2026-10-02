# API reference

Generated from docstrings. Everything listed is importable from the top-level
`fhirground` namespace unless noted.

## Pipeline

::: fhirground.pipeline
    options:
      members: [Pipeline, PipelineConfig, NoteResult, MentionTrace, run_notes, summarize]

## Input

::: fhirground.io
    options:
      members: [NoteInput, read_notes, read_notes_jsonl, read_notes_fhir, write_notes_jsonl]

## The mention schema

::: fhirground.schemas.mention
    options:
      members: [Mention, MentionList, MentionType, Assertion, ClinicalStatusHint, MedAction, ObsCategory, ValueComparator, assert_no_code_fields]

::: fhirground.schemas.jsonschema
    options:
      members: [build_mention_schema, load_mention_schema, write_mention_schema, assert_backend_safe, schema_digest]

## Preprocessing

::: fhirground.preprocess.sections
    options:
      members: [Section, detect_sections, section_at, is_non_patient_section]

::: fhirground.preprocess.chunking
    options:
      members: [Chunk, ChunkPlan, chunk_note]

## Backends

::: fhirground.backends.base
    options:
      members: [Backend, RawResponse]

::: fhirground.backends.mock
    options:
      members: [MockBackend, extract_by_rules, record_fixtures]

::: fhirground.backends.openai_compatible
    options:
      members: [OpenAICompatible]

::: fhirground.backends.bedrock
    options:
      members: [BedrockBackend]

::: fhirground.backends.prompts
    options:
      members: [render, render_repair, RenderedPrompt]

## Extraction and the repair loop

::: fhirground.extract
    options:
      members: [extract_chunk, ChunkExtraction, parse_mentions, strip_to_json, CallCache]

## Grounding

::: fhirground.grounding
    options:
      members: [ground_mention, ground_in_chunk, Grounding, GroundingStrategy, normalize]

## The assertion policy

::: fhirground.assertion
    options:
      members: [decide, EmissionDecision, DropReason, VerificationStatus, RequestStatus]

## Normalization

::: fhirground.normalize.units
    options:
      members: [resolve_unit, UnitResolution, is_known_ucum_code, known_unit_count]

::: fhirground.normalize.values
    options:
      members: [parse_value, parse_dose, ParsedValue]

::: fhirground.normalize.dates
    options:
      members: [resolve_date, inferred_from_note, ResolvedDate, DatePrecision]

## Terminology

::: fhirground.terminology.base
    options:
      members: [TerminologyResolver, Resolution, Coding, ResolverConfig]

::: fhirground.terminology.local
    options:
      members: [LocalValueSet, NullResolver, ValueSetEntry]

::: fhirground.terminology.chain
    options:
      members: [Chain]

::: fhirground.terminology.rxnav
    options:
      members: [RxNavResolver]

## Deduplication

::: fhirground.dedup
    options:
      members: [dedup_mentions, dedup_key, MentionGroup, matches_for_consistency]

## Mapping

::: fhirground.mapping.condition
    options:
      members: [build_condition]

::: fhirground.mapping.medication_request
    options:
      members: [build_medication_request]

::: fhirground.mapping.observation
    options:
      members: [build_observation]

::: fhirground.mapping.provenance
    options:
      members: [build_provenance, build_bundle]

::: fhirground.mapping.extensions
    options:
      members: [ExtensionUrls, confidence_extension, evidence_extension, effective_inferred_extension, read_confidence, read_calibrated, read_evidence]

## Validation

::: fhirground.validate
    options:
      members: [validate_resource, validate_bundle, parse_with_models, check_extensions, check_evidence_offsets, check_codings_have_system_and_code, check_ucum_usage, check_allowed_fields, check_fhir_invariants, run_hl7_validator, ValidationReport, Finding, Severity]

## Confidence

::: fhirground.confidence
    options:
      members: [ConfidenceComponents, aggregate, validation_component, self_consistency, Calibrator]

## Output

::: fhirground.export
    options:
      members: [write_outputs, write_bundles, write_ndjson, write_clinval_predictions, write_debug, write_summary, clinval_label]

## Scoring

::: fhirground.evaluate
    options:
      members: [score_resources, resources_match, read_resources, MatchSpec, ScoreReport, TypeScore]

## Synthetic data

::: fhirground.synth.notes_from_synthea
    options:
      members: [build_benchmark, write_benchmark, generate_records, records_from_synthea]

::: fhirground.synth.templates
    options:
      members: [SourceRecord, RenderedNote, render_note, value_sets_from_records]

## Exceptions

::: fhirground.exceptions
    options:
      members: [FhirgroundError, InputError, ConfigError, BackendError, SchemaValidationError, GroundingError, TerminologyError, MappingError, ValidationRuleError, CalibrationError]
