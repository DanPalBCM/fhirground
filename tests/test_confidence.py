"""Confidence components, renormalization, and the calibrator."""

from __future__ import annotations

import json

import pytest

from fhirground.confidence import (
    DEFAULT_WEIGHTS,
    Calibrator,
    ConfidenceComponents,
    aggregate,
    self_consistency,
    validation_component,
)
from fhirground.exceptions import CalibrationError


class TestComponents:
    def test_available_excludes_none(self):
        c = ConfidenceComponents(g=1.0, c=None, t=0.9, v=1.0, l=None)
        assert set(c.available()) == {"g", "t", "v"}

    def test_as_dict_keeps_the_absent_ones(self):
        """Absent must stay distinguishable from zero in the debug output."""
        d = ConfidenceComponents(g=1.0).as_dict()
        assert d["g"] == 1.0
        assert d["c"] is None
        assert set(d) == {"g", "c", "t", "v", "l"}

    @pytest.mark.parametrize(("repairs", "expected"), [(0, 1.0), (1, 0.8), (2, 0.6), (5, 0.6)])
    def test_validation_component(self, repairs, expected):
        assert validation_component(repairs) == expected

    def test_negative_repairs_rejected(self):
        with pytest.raises(ValueError, match="cannot be negative"):
            validation_component(-1)


class TestAggregate:
    def test_single_component_is_itself(self):
        assert aggregate(ConfidenceComponents(g=0.9)) == pytest.approx(0.9)

    def test_weights_renormalize_over_available(self):
        """With no terminology, g and v share the full weight."""
        c = ConfidenceComponents(g=1.0, v=0.6)
        expected = (1.0 * 0.35 + 0.6 * 0.15) / (0.35 + 0.15)
        assert aggregate(c) == pytest.approx(expected)

    def test_missing_component_does_not_act_as_zero(self):
        """A NullResolver must not be punished as if terminology failed."""
        without = aggregate(ConfidenceComponents(g=1.0, v=1.0))
        with_zero = aggregate(ConfidenceComponents(g=1.0, v=1.0, t=0.0))
        assert without > with_zero
        assert without == pytest.approx(1.0)

    def test_all_components(self):
        c = ConfidenceComponents(g=1.0, c=1.0, t=1.0, v=1.0, l=1.0)
        assert aggregate(c) == pytest.approx(1.0)

    def test_no_components_is_zero(self):
        assert aggregate(ConfidenceComponents()) == 0.0

    def test_custom_weights(self):
        c = ConfidenceComponents(g=1.0, t=0.0)
        assert aggregate(c, {"g": 1.0, "t": 0.0}) == pytest.approx(1.0)
        assert aggregate(c, {"g": 0.0, "t": 1.0}) == pytest.approx(0.0)

    def test_negative_weights_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            aggregate(ConfidenceComponents(g=1.0), {"g": -1.0})

    def test_all_zero_weights_on_available_components_is_an_error(self):
        with pytest.raises(ValueError, match="undefined"):
            aggregate(ConfidenceComponents(l=1.0), DEFAULT_WEIGHTS)

    def test_result_is_in_range(self):
        for g in (0.0, 0.5, 1.0):
            for t in (0.0, 0.5, 1.0):
                assert 0.0 <= aggregate(ConfidenceComponents(g=g, t=t)) <= 1.0

    def test_default_weights_match_the_spec(self):
        assert DEFAULT_WEIGHTS == {"g": 0.35, "c": 0.30, "t": 0.20, "v": 0.15, "l": 0.0}


class TestSelfConsistency:
    def test_fraction(self):
        assert self_consistency(2, 4) == 0.5
        assert self_consistency(3, 3) == 1.0

    def test_single_sample_is_none(self):
        """One sample has nothing to be consistent with."""
        assert self_consistency(1, 1) is None
        assert self_consistency(0, 0) is None

    def test_out_of_range_rejected(self):
        with pytest.raises(ValueError, match=r"\[0, 3\]"):
            self_consistency(5, 3)


class TestCalibrator:
    @pytest.fixture
    def data(self):
        scores = [i / 50 for i in range(50)]
        labels = [0 if s < 0.5 else 1 for s in scores]
        return scores, labels

    def test_unfitted_transform_raises(self):
        """Returning the input would let a resource claim calibrated: true."""
        with pytest.raises(CalibrationError, match="not been fitted"):
            Calibrator().transform(0.5)

    def test_isotonic_fit_and_transform(self, data):
        scores, labels = data
        cal = Calibrator().fit(scores, labels, method="isotonic")
        assert cal.fitted
        assert cal.n_samples == len(scores)
        assert 0.0 <= cal.transform(0.1) <= 1.0
        assert cal.transform(0.9) > cal.transform(0.1)

    def test_isotonic_is_monotone(self, data):
        scores, labels = data
        cal = Calibrator().fit(scores, labels, method="isotonic")
        outputs = [cal.transform(x / 20) for x in range(21)]
        assert outputs == sorted(outputs)

    def test_platt_fit_and_transform(self, data):
        scores, labels = data
        cal = Calibrator().fit(scores, labels, method="platt")
        assert cal.coef is not None
        assert cal.transform(0.9) > cal.transform(0.1)

    def test_platt_is_monotone(self, data):
        scores, labels = data
        cal = Calibrator().fit(scores, labels, method="platt")
        outputs = [cal.transform(x / 20) for x in range(21)]
        assert outputs == sorted(outputs)

    def test_outputs_are_clipped_to_unit_interval(self, data):
        scores, labels = data
        for method in ("isotonic", "platt"):
            cal = Calibrator().fit(scores, labels, method=method)
            for x in (-5.0, 0.0, 0.5, 1.0, 5.0):
                assert 0.0 <= cal.transform(x) <= 1.0

    def test_transform_many(self, data):
        scores, labels = data
        cal = Calibrator().fit(scores, labels)
        assert len(cal.transform_many([0.1, 0.5, 0.9])) == 3

    def test_too_little_data_rejected(self):
        with pytest.raises(CalibrationError, match="at least 10"):
            Calibrator().fit([0.1, 0.9], [0, 1])

    def test_single_class_rejected(self):
        """One class says nothing about where the boundary is."""
        with pytest.raises(CalibrationError, match="single class"):
            Calibrator().fit([i / 20 for i in range(20)], [1] * 20)

    def test_length_mismatch_rejected(self):
        with pytest.raises(CalibrationError, match="differ in length"):
            Calibrator().fit([0.1] * 12, [1] * 11)

    def test_unknown_method_rejected(self, data):
        scores, labels = data
        with pytest.raises(CalibrationError, match="unknown calibration method"):
            Calibrator().fit(scores, labels, method="magic")

    @pytest.mark.parametrize("method", ["isotonic", "platt"])
    def test_json_round_trip(self, data, tmp_path, method):
        scores, labels = data
        cal = Calibrator().fit(scores, labels, method=method)
        path = cal.save(tmp_path / "c.json")
        loaded = Calibrator.load(path)
        for x in (0.0, 0.25, 0.5, 0.75, 1.0):
            assert loaded.transform(x) == pytest.approx(cal.transform(x))

    def test_saved_file_is_json_not_pickle(self, data, tmp_path):
        """A pickled model is code execution on load."""
        scores, labels = data
        path = Calibrator().fit(scores, labels).save(tmp_path / "c.json")
        payload = json.loads(path.read_text())
        assert payload["format"] == "fhirground-calibrator/1"
        assert payload["n_samples"] == len(scores)

    def test_load_missing_file(self, tmp_path):
        with pytest.raises(CalibrationError, match="not found"):
            Calibrator.load(tmp_path / "nope.json")

    def test_load_malformed(self, tmp_path):
        path = tmp_path / "c.json"
        path.write_text("{oops")
        with pytest.raises(CalibrationError, match="malformed JSON"):
            Calibrator.load(path)

    def test_load_wrong_shape(self, tmp_path):
        path = tmp_path / "c.json"
        path.write_text(json.dumps({"hello": "world"}))
        with pytest.raises(CalibrationError, match="not a fhirground calibrator"):
            Calibrator.load(path)
