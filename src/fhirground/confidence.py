"""Confidence scoring and calibration.

**An uncalibrated score is not a probability.** It is a weighted mean of
heuristics, and 0.8 does not mean "right 80% of the time". Every resource
records whether its score was calibrated, the docs say this in the same words,
and :class:`Calibrator` exists so the claim can be made true on labelled data.

Five components (SPEC §4.6), each in [0, 1] or absent:

===========  ======  ====================================================
component    weight  meaning
===========  ======  ====================================================
``g``          .35   grounding: 1.0 exact, 0.9 normalized, ratio * 0.8 fuzzy
``c``          .30   self-consistency across ``k`` samples
``t``          .20   the resolver's match score
``v``          .15   validation: 1.0 no repairs, 0.8 one, 0.6 two
``l``          .00   mean token probability, when the backend exposes it
===========  ======  ====================================================

Weights are **renormalized over the components that are actually available**,
which is the part that matters: with a NullResolver there is no ``t``, and
scoring it as 0 would punish every resource for a configuration choice rather
than for anything about the extraction.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fhirground.exceptions import CalibrationError

#: Default component weights (SPEC §4.6).
DEFAULT_WEIGHTS: dict[str, float] = {"g": 0.35, "c": 0.30, "t": 0.20, "v": 0.15, "l": 0.0}

#: Validation component by repair count. Beyond two, the floor holds.
_REPAIR_SCORES = (1.0, 0.8, 0.6)


@dataclass(frozen=True, slots=True)
class ConfidenceComponents:
    """The per-mention components, before aggregation.

    ``None`` means *unavailable*, which is different from 0.0 meaning *bad*.
    Keeping them apart is the whole reason this is a dataclass of optionals
    rather than a dict of floats.
    """

    g: float | None = None
    c: float | None = None
    t: float | None = None
    v: float | None = None
    l: float | None = None  # noqa: E741 - the spec names this component `l`

    def available(self) -> dict[str, float]:
        """The components that have values."""
        return {
            k: v
            for k, v in (("g", self.g), ("c", self.c), ("t", self.t), ("v", self.v), ("l", self.l))
            if v is not None
        }

    def as_dict(self) -> dict[str, float | None]:
        """All five components, including the absent ones, for the debug trace."""
        return {"g": self.g, "c": self.c, "t": self.t, "v": self.v, "l": self.l}


def validation_component(repairs: int) -> float:
    """Validation component from the number of repair attempts used."""
    if repairs < 0:
        raise ValueError(f"repairs cannot be negative, got {repairs}")
    return _REPAIR_SCORES[min(repairs, len(_REPAIR_SCORES) - 1)]


def aggregate(components: ConfidenceComponents, weights: dict[str, float] | None = None) -> float:
    """Weighted mean over the available components, renormalized.

    Returns:
        The score in [0, 1]. When no component is available, returns 0.0 --
        a resource nobody could say anything about gets no confidence, which
        ``min_confidence`` can then filter.

    Raises:
        ValueError: If a weight is negative, or every available component has
            weight zero (which would make the score undefined rather than 0).
    """
    w = dict(weights or DEFAULT_WEIGHTS)
    if any(v < 0 for v in w.values()):
        raise ValueError(f"weights must be non-negative, got {w}")

    present = components.available()
    if not present:
        return 0.0

    total = sum(w.get(k, 0.0) for k in present)
    if total <= 0:
        raise ValueError(
            f"every available component {sorted(present)} has weight 0 under {w}, so the "
            "score would be undefined. Give at least one of them a positive weight."
        )
    return round(sum(present[k] * w.get(k, 0.0) for k in present) / total, 6)


def self_consistency(matches: int, samples: int) -> float | None:
    """Fraction of samples that produced a matching mention.

    Returns None when ``samples <= 1``: with one sample there is nothing to be
    consistent with, and reporting 1.0 would claim agreement that was never
    tested.
    """
    if samples <= 1:
        return None
    if matches < 0 or matches > samples:
        raise ValueError(f"matches ({matches}) must be in [0, {samples}]")
    return round(matches / samples, 6)


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------


@dataclass
class Calibrator:
    """Maps heuristic scores to calibrated probabilities.

    Two methods, both saved as **JSON, never pickle** -- a pickled model is
    arbitrary code execution on load, which is not an acceptable thing to put
    in a health data pipeline.

    * ``isotonic`` -- a monotone step function. Flexible and needs no
      distributional assumption, but can overfit on small data.
    * ``platt`` -- a one-parameter logistic fit. Smoother and more robust when
      labelled data is scarce, which it usually is.

    Attributes:
        method: ``isotonic`` or ``platt``.
        thresholds: Isotonic x-values.
        values: Isotonic y-values.
        coef: Platt slope.
        intercept: Platt intercept.
        n_samples: How much labelled data the fit used, recorded because a
            calibrator fitted on thirty examples deserves less trust than one
            fitted on three thousand.
    """

    method: str = "isotonic"
    thresholds: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    coef: float | None = None
    intercept: float | None = None
    n_samples: int = 0
    fitted: bool = False

    def fit(
        self, scores: list[float], labels: list[int | bool], method: str | None = None
    ) -> Calibrator:
        """Fit on labelled dev data.

        Args:
            scores: Uncalibrated scores.
            labels: 1 for a correct resource, 0 otherwise.
            method: ``isotonic`` or ``platt``. Defaults to the instance's.

        Raises:
            CalibrationError: On mismatched lengths, too little data, or a
                single-class label set. One class carries no information about
                where the decision boundary is, so refusing beats returning a
                constant that looks like a probability.
        """
        chosen = (method or self.method).lower()
        if chosen not in ("isotonic", "platt"):
            raise CalibrationError(
                f"unknown calibration method {chosen!r}; use 'isotonic' or 'platt'"
            )
        if len(scores) != len(labels):
            raise CalibrationError(
                f"scores and labels differ in length ({len(scores)} vs {len(labels)})"
            )
        if len(scores) < 10:
            raise CalibrationError(
                f"need at least 10 labelled examples to fit a calibrator, got {len(scores)}"
            )
        ints = [int(bool(y)) for y in labels]
        if len(set(ints)) < 2:
            raise CalibrationError(
                "labels contain a single class, so a calibrator cannot be fitted. "
                "Calibration needs both correct and incorrect examples."
            )

        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - extra is declared
            raise CalibrationError(
                "calibration needs the [calibrate] extra: pip install 'fhirground[calibrate]'"
            ) from exc

        x = np.asarray(scores, dtype=float)
        y = np.asarray(ints, dtype=float)

        if chosen == "isotonic":
            try:
                from sklearn.isotonic import IsotonicRegression
            except ImportError as exc:  # pragma: no cover
                raise CalibrationError(
                    "isotonic calibration needs scikit-learn: pip install 'fhirground[calibrate]'"
                ) from exc
            model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            model.fit(x, y)
            self.thresholds = [float(v) for v in model.X_thresholds_]
            self.values = [float(v) for v in model.y_thresholds_]
            self.coef = None
            self.intercept = None
        else:
            try:
                from sklearn.linear_model import LogisticRegression
            except ImportError as exc:  # pragma: no cover
                raise CalibrationError(
                    "Platt scaling needs scikit-learn: pip install 'fhirground[calibrate]'"
                ) from exc
            model = LogisticRegression(solver="lbfgs")
            model.fit(x.reshape(-1, 1), y)
            self.coef = float(model.coef_[0][0])
            self.intercept = float(model.intercept_[0])
            self.thresholds = []
            self.values = []

        self.method = chosen
        self.n_samples = len(scores)
        self.fitted = True
        return self

    def transform(self, score: float) -> float:
        """Map one score to a calibrated probability.

        Raises:
            CalibrationError: If the calibrator was never fitted. Returning
                the input unchanged would let a resource claim
                ``calibrated: true`` while carrying a raw heuristic.
        """
        if not self.fitted:
            raise CalibrationError("calibrator has not been fitted; call fit() or load() first")
        if self.method == "isotonic":
            return round(_interp(score, self.thresholds, self.values), 6)
        if self.coef is None or self.intercept is None:  # pragma: no cover
            raise CalibrationError("Platt calibrator is missing its parameters")
        z = self.coef * score + self.intercept
        return round(1.0 / (1.0 + math.exp(-z)), 6)

    def transform_many(self, scores: list[float]) -> list[float]:
        """Map a list of scores."""
        return [self.transform(s) for s in scores]

    # -- persistence ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe representation."""
        return {
            "format": "fhirground-calibrator/1",
            "method": self.method,
            "thresholds": self.thresholds,
            "values": self.values,
            "coef": self.coef,
            "intercept": self.intercept,
            "n_samples": self.n_samples,
            "fitted": self.fitted,
        }

    def save(self, path: str | Path) -> Path:
        """Write as JSON. No pickle, by design."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2))
        return target

    @classmethod
    def load(cls, path: str | Path) -> Calibrator:
        """Load from JSON.

        Raises:
            CalibrationError: If the file is missing, malformed, or not a
                fhirground calibrator.
        """
        source = Path(path)
        if not source.exists():
            raise CalibrationError(f"calibrator file not found: {source}")
        try:
            data = json.loads(source.read_text())
        except json.JSONDecodeError as exc:
            raise CalibrationError(f"{source}: malformed JSON: {exc.msg}") from exc
        if not isinstance(data, dict) or "method" not in data:
            raise CalibrationError(f"{source} is not a fhirground calibrator file")
        return cls(
            method=str(data.get("method", "isotonic")),
            thresholds=[float(v) for v in data.get("thresholds") or []],
            values=[float(v) for v in data.get("values") or []],
            coef=None if data.get("coef") is None else float(data["coef"]),
            intercept=None if data.get("intercept") is None else float(data["intercept"]),
            n_samples=int(data.get("n_samples") or 0),
            fitted=bool(data.get("fitted", True)),
        )


def _interp(x: float, xs: list[float], ys: list[float]) -> float:
    """Piecewise-linear interpolation with clipping outside the fitted range."""
    if not xs or not ys:
        raise CalibrationError("isotonic calibrator has no thresholds")
    if x <= xs[0]:
        return max(0.0, min(1.0, ys[0]))
    if x >= xs[-1]:
        return max(0.0, min(1.0, ys[-1]))
    for i in range(1, len(xs)):
        if x <= xs[i]:
            x0, x1 = xs[i - 1], xs[i]
            y0, y1 = ys[i - 1], ys[i]
            if x1 == x0:
                return max(0.0, min(1.0, y1))
            frac = (x - x0) / (x1 - x0)
            return max(0.0, min(1.0, y0 + frac * (y1 - y0)))
    return max(0.0, min(1.0, ys[-1]))  # pragma: no cover
