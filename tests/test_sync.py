import numpy as np

from src import sync


def test_estimate_offset_recovers_known_shift():
    rng = np.random.default_rng(0)
    beats = np.cumsum(rng.uniform(0.75, 0.95, 30))
    shift = 1.4
    off, score = sync.estimate_offset(beats, beats + shift)
    assert abs(off - shift) < 0.1
    assert score > 0.7


def test_estimate_offset_survives_missing_beats():
    rng = np.random.default_rng(1)
    beats = np.cumsum(rng.uniform(0.75, 0.95, 40))
    partial = np.delete(beats, rng.choice(40, 12, replace=False)) + 0.8
    off, score = sync.estimate_offset(beats, partial)
    assert abs(off - 0.8) < 0.15
    assert score > 0.4


def test_estimate_offset_reports_low_score_on_unrelated_sequences():
    rng = np.random.default_rng(2)
    a = np.cumsum(rng.uniform(0.7, 1.0, 30))
    b = np.cumsum(rng.uniform(0.4, 0.5, 30))
    off, score = sync.estimate_offset(a, b)
    assert score < 0.5


def test_estimate_offset_returns_zero_score_on_empty_input():
    assert sync.estimate_offset(np.zeros(0), np.arange(10.0)) == (0.0, 0.0)


def test_estimate_offset_score_is_bounded():
    rng = np.random.default_rng(3)
    beats = np.cumsum(rng.uniform(0.75, 0.95, 25))
    _, score = sync.estimate_offset(beats, beats)
    assert 0.0 <= score <= 1.0
    assert score > 0.95  # сам с собой совпадает почти полностью
