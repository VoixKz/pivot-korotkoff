import numpy as np
import torch

from src import config, dataset


def test_interval_to_mask_marks_the_right_frames():
    m = dataset.interval_to_mask(2.0, 4.0, n_frames=500)  # 10 с при шаге 20 мс
    assert m.shape == (500,)
    assert m[:100].sum() == 0
    assert m[100:200].min() == 1.0
    assert m[200:].sum() == 0


def test_mask_to_interval_is_inverse_of_interval_to_mask():
    m = dataset.interval_to_mask(3.0, 7.5, n_frames=600)
    s, e = dataset.mask_to_interval(m)
    assert abs(s - 3.0) < config.HOP_S * 1.5
    assert abs(e - 7.5) < config.HOP_S * 1.5


def test_mask_to_interval_returns_none_when_empty():
    assert dataset.mask_to_interval(np.zeros(100, dtype=np.float32)) is None


def test_interval_to_mask_clamps_out_of_range_values():
    m = dataset.interval_to_mask(-5.0, 999.0, n_frames=50)
    assert m.sum() == 50


def test_collate_pads_and_reports_lengths():
    a = (torch.randn(100, config.N_FEATURES), torch.rand(100), 1)
    b = (torch.randn(60, config.N_FEATURES), torch.rand(60), 2)
    x, y, lens = dataset.collate([a, b])
    assert x.shape == (2, 100, config.N_FEATURES)
    assert y.shape == (2, 100)
    assert lens.tolist() == [100, 60]
    assert torch.all(x[1, 60:] == 0)


def test_read_labels_parses_the_autolabel_csv(tmp_path):
    f = tmp_path / "l.csv"
    f.write_text("id,start,end,confidence,priority,source,note\n"
                 "5,1.25,9.75,0.4,0.6,audio,\n")
    lab = dataset.read_labels(f)
    assert lab[5] == (1.25, 9.75)


def test_augmentation_keeps_mask_aligned_with_features():
    """Аугментация меняет длину сигнала — метка обязана поехать вместе с ним."""
    from tests.test_autolabel import _korotkoff

    x = _korotkoff(dur=20.0, start=6.0, end=14.0)
    for shift in (-1.5, 0.0, 2.0):
        y, s, e = dataset.augment_signal(x, 6.0, 14.0, shift=shift, stretch=1.0,
                                         gain=1.0, noise=0.0)
        assert abs(s - (6.0 + shift)) < 0.05
        assert abs(e - (14.0 + shift)) < 0.05
    y, s, e = dataset.augment_signal(x, 6.0, 14.0, shift=0.0, stretch=1.1,
                                     gain=1.0, noise=0.0)
    assert abs(len(y) / config.SR - 22.0) < 0.2
    assert abs(s - 6.6) < 0.1 and abs(e - 15.4) < 0.1
