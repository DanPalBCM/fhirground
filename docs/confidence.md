# Confidence and calibration

## An uncalibrated score is not a probability

It is a weighted mean of heuristics. **0.8 does not mean "right 80% of the
time."**

Every resource records whether its score was calibrated:

```json
{"url": ".../extraction-confidence", "valueDecimal": 0.87,
 "extension": [{"url": "calibrated", "valueBoolean": false}]}
```

That flag is the difference between a probability and a heuristic, which is why
it is on every resource rather than only in the run summary.

## The components

| Component | Weight | Meaning |
| --- | --- | --- |
| `g` grounding | 0.35 | 1.0 exact, 0.9 normalized, `ratio/100 * 0.8` fuzzy |
| `c` self-consistency | 0.30 | fraction of `k` samples producing a matching mention |
| `t` terminology | 0.20 | the resolver's match score |
| `v` validation | 0.15 | 1.0 with no repairs, 0.8 after one, 0.6 after two |
| `l` log-probability | 0.00 | mean token probability, when the backend exposes it |

## Absent is not zero

The part that matters. Weights **renormalize over the components that are
actually available**:

```python
# With a NullResolver there is no `t`. Scoring it 0 would punish every
# resource for a configuration choice rather than for anything observed.
aggregate(ConfidenceComponents(g=1.0, v=1.0))        # 1.0
aggregate(ConfidenceComponents(g=1.0, v=1.0, t=0.0)) # 0.72 -- a real miss
```

`c` is `None` when `samples == 1`: with one sample there is nothing to be
consistent with, and reporting 1.0 would claim agreement that was never tested.
Setting `samples > 1` at temperature 0 is rejected at config time for the same
reason — identical samples would make the score meaninglessly 1.0.

All five components, including the absent ones, are written to `debug.jsonl`.

## Calibration

```bash
# 1. run on labelled dev data
fhirground extract --input dev_notes.jsonl --value-sets valuesets/ --out dev_out/

# 2. fit, using gold to label each emitted resource
fhirground calibrate --debug dev_out/debug.jsonl --gold dev_gold/ \
  --method isotonic --out calibrator.json

# 3. apply
fhirground extract --input notes.jsonl --calibrator calibrator.json --out out/
```

Resources now carry `calibrated: true` and the score is a probability estimate.

| Method | When |
| --- | --- |
| `isotonic` | monotone step function; flexible, no distributional assumption, can overfit on small data |
| `platt` | one-parameter logistic; smoother and more robust when labelled data is scarce, which it usually is |

### Guards

- **At least 10 labelled examples.** Fewer is refused.
- **Both classes required.** A single-class label set carries no information
  about where the boundary is, so fitting is refused rather than returning a
  constant that looks like a probability.
- **`transform` on an unfitted calibrator raises.** Returning the input
  unchanged would let a resource claim `calibrated: true` while carrying a raw
  heuristic.
- **`n_samples` is recorded** in the saved file, because a calibrator fitted on
  thirty examples deserves less trust than one fitted on three thousand.

### Saved as JSON, never pickle

A pickled model is arbitrary code execution on load. That is not an acceptable
thing to put in a health data pipeline, so `Calibrator.save` writes JSON:

```json
{"format": "fhirground-calibrator/1", "method": "isotonic",
 "thresholds": [...], "values": [...], "n_samples": 1240, "fitted": true}
```

## Filtering

`min_confidence` drops resources below a threshold, counted as
`below_threshold`. It defaults to 0.0 — filtering on an **uncalibrated** score
is choosing a cut-point on an arbitrary scale, so it should be a deliberate
decision, ideally after calibration.
