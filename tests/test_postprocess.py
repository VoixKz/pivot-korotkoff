import numpy as np

from src import config
from src import postprocess as pp


def test_close_gaps_bridges_short_hole():
    """Аускультативный провал: тоны пропадают на 2 с и возвращаются."""
    m = np.zeros(500, dtype=bool)
    m[100:200] = True
    m[300:400] = True  # дыра 100 кадров = 2.0 с
    out = pp.close_gaps(m, max_gap_s=3.0)
    assert out[200:300].all()
    assert not out[:100].any() and not out[400:].any()


def test_close_gaps_keeps_long_hole_open():
    m = np.zeros(600, dtype=bool)
    m[50:100] = True
    m[400:450] = True  # дыра 300 кадров = 6.0 с
    assert not pp.close_gaps(m, max_gap_s=3.0)[200:300].any()


def test_longest_run_picks_the_longest():
    m = np.zeros(100, dtype=bool)
    m[5:10] = True
    m[40:70] = True
    assert pp.longest_run(m) == (40, 70)


def test_longest_run_returns_none_on_empty_mask():
    assert pp.longest_run(np.zeros(50, dtype=bool)) is None


def test_to_interval_recovers_a_clean_block():
    prob = np.zeros(500, dtype=np.float32)
    prob[150:350] = 0.9
    r = pp.to_interval(prob)
    assert abs(r["start"] - 150 * config.HOP_S) < 0.1
    assert abs(r["end"] - 350 * config.HOP_S) < 0.1
    assert r["confidence"] > 0.8


def test_to_interval_returns_none_when_nothing_crosses_threshold():
    assert pp.to_interval(np.full(300, 0.1, dtype=np.float32)) is None


def test_to_interval_bridges_an_auscultatory_gap():
    prob = np.zeros(600, dtype=np.float32)
    prob[100:250] = 0.9
    prob[330:480] = 0.9  # провал 1.6 с
    r = pp.to_interval(prob, max_gap_s=3.0)
    assert abs(r["start"] - 100 * config.HOP_S) < 0.15
    assert abs(r["end"] - 480 * config.HOP_S) < 0.15


def test_refine_edge_gives_subframe_precision():
    prob = np.zeros(200, dtype=np.float32)
    prob[100:] = 1.0
    prob[99] = 0.5
    t = pp.refine_edge(prob, 100, threshold=0.5)
    assert 99 * config.HOP_S <= t <= 100 * config.HOP_S


def test_smooth_removes_single_frame_spikes():
    prob = np.zeros(200, dtype=np.float32)
    prob[100] = 1.0
    assert pp.smooth(prob, win_s=0.5).max() < 0.5
