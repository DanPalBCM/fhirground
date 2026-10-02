# Backends

A backend turns a prompt plus the mention schema into raw text. It does **not**
parse, validate, repair, or know about FHIR — `fhirground.extract` does that.
Keeping the boundary there means a new backend is one small class and the repair
loop is tested once rather than per backend.

## mock

Deterministic, offline, fixture-driven. Used by every unit test and by the
demo, which is how the whole pipeline runs in CI with no network and no API key.

```python
from fhirground.backends import MockBackend
MockBackend()                              # rule extractor
MockBackend.from_fixture_file("f.json")    # recorded responses
MockBackend(fail_first_n=1)                # exercise the repair loop
```

The rule extractor is deliberately simple-minded and makes the mistakes a weak
model makes — it reads cues sentence-wide, so a sentence containing both a
negation and a second finding mislabels the second. That is on purpose: a
harness whose own tests only ever see clean input has not been tested.

## openai_compatible (vLLM and friends)

```bash
fhirground extract --backend openai_compatible \
  --base-url http://localhost:8000/v1 --model meta-llama/Llama-3.1-8B-Instruct ...
```

Request shape **verified** against the OpenAI client's own type definitions:

```json
{"response_format": {"type": "json_schema",
  "json_schema": {"name": "MentionList", "schema": {...}, "strict": false}}}
```

Two constraint modes, because servers differ:

- `json_schema` (default) — the OpenAI-compatible spelling, which recent vLLM
  supports and which gives true token-level constrained decoding when the
  server's guided backend is active.
- `guided_json` — vLLM's older extra-body parameter, for servers predating the
  former.
- `none` — for a server with no constrained decoding at all. Expect to lean on
  the repair loop.

## bedrock

```bash
fhirground extract --backend bedrock --model "$MODEL_ID" --region us-east-1 ...
```

Credentials come from the standard boto3 chain. **fhirground never holds AWS
keys.**

Request shape **verified** against botocore's own service model for
`bedrock-runtime`: Converse has no `response_format`-style structured output,
so fhirground uses a single **forced tool** whose `inputSchema.json` is the
mention schema, with `toolChoice: {tool: {name: ...}}`. The model's tool
arguments *are* the mention JSON.

If the model answers in prose anyway, the text is passed through so the repair
loop can tell it what was wrong — rather than raising a backend error for
something the model can fix.

Errors are translated into actionable messages: an auth failure names the boto3
chain and the region, throttling suggests lowering `--concurrency`, and a
`ValidationException` mentioning the tool schema points at
`assert_backend_safe`.

## Why `strict` is off by default on both

OpenAI's strict mode requires every property to appear in `required`. The
mention schema has four required fields and thirteen optional ones **on
purpose** — a model forced to fill every field invents values for the ones it
cannot support. vLLM's guided decoding constrains to the schema without that
requirement. So strict buys nothing here and costs correctness. It remains
configurable.

## The repair loop

On a JSON or schema failure, fhirground re-prompts with the validation error,
up to `max_repairs` (default 2). The error message is written to be read **by a
model** — it names the offending field and what was expected:

```
The JSON did not match the schema:
- mentions.0.snomed_code: this field is not in the schema. Remove it.
  Remember that there are no code fields.
```

The repair count feeds the `v` confidence component, so a mention that took two
attempts is worth less than one that parsed first time.

A **transport or auth failure is not a repair case.** `BackendError` propagates
immediately rather than burning attempts on something that was never the
model's fault.

Truncation is surfaced: `finish_reason == "length"` means incomplete JSON, and
the repair prompt says so, because otherwise the model would be asked to fix a
problem it cannot see.

## Writing your own

```python
class MyBackend:
    name = "mine"
    model_id = "my-model"

    def extract(self, prompt, schema, *, temperature=0.0, n=1):
        ...  # return [RawResponse(text=...)]

    def describe(self):
        return {"backend": self.name, "model_id": self.model_id}
```

That is the whole protocol. Populate `tokens_in`, `tokens_out`, `latency_s`,
and `finish_reason` when your provider reports them — they feed the cost
summary and the truncation warning. `describe()` goes into `run_summary.json`
and the cache key, so include anything that changes the output.

## The call cache

```bash
fhirground extract --cache .cache/llm.sqlite ...
```

Keyed by a hash of backend description, prompt, temperature, sample count, and
schema digest, so changing any of them is a miss rather than a stale hit.

!!! warning "The cache contains note text"
    The cached value includes the prompt, and the prompt includes the chunk. It
    is note-derived data subject to the same handling rules as the notes. Off
    by default. See [PHI](phi.md).
