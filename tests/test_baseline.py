import numpy as np

from src import baseline, config
from tests.test_autolabel import _korotkoff


def test_baseline_finds_beats_in_the_middle():
    r = baseline.predict_array(_korotkoff(dur=26.0, start=8.0, end=18.0))
    assert r["start"] is not None
    assert abs(r["start"] - 8.0) < 2.5
    assert abs(r["end"] - 18.0) < 2.5


def test_baseline_reports_low_confidence_on_pure_noise():
    x = np.random.default_rng(1).normal(0, 0.1, config.SR * 20).astype(np.float32)
    assert baseline.predict_array(x)["confidence"] < 0.5


def test_baseline_never_returns_reversed_interval():
    r = baseline.predict_array(_korotkoff(dur=20.0, start=4.0, end=15.0, seed=7))
    assert r["end"] > r["start"]


def test_baseline_output_has_the_same_keys_as_the_model():
    r = baseline.predict_array(_korotkoff())
    assert set(r) >= {"start", "end", "confidence", "duration"}
