import json
import subprocess
import sys

import numpy as np
import torch

from src import config, model, predict


def _fresh_checkpoint(tmp_path):
    net = model.KorotkoffNet()
    p = tmp_path / "ckpt.pt"
    torch.save({"state_dict": net.state_dict(), "iou": 0.0, "epoch": 0}, p)
    return p


def test_frame_probabilities_are_in_unit_range(tmp_path):
    from tests.test_autolabel import _korotkoff

    ckpt = _fresh_checkpoint(tmp_path)
    predict._CACHE.clear()
    x = _korotkoff(dur=12.0)
    prob = predict.probabilities_from_array(x, ckpt)
    assert prob.ndim == 1
    assert prob.min() >= 0.0 and prob.max() <= 1.0
    assert len(prob) == len(x) // int(config.HOP_S * config.SR)


def test_predict_array_returns_the_agreed_shape(tmp_path):
    from tests.test_autolabel import _korotkoff

    ckpt = _fresh_checkpoint(tmp_path)
    predict._CACHE.clear()
    r = predict.predict_array(_korotkoff(dur=12.0), ckpt)
    assert set(r) >= {"start", "end", "confidence", "duration"}
    assert abs(r["duration"] - 12.0) < 0.1
    if r["start"] is not None:
        assert 0.0 <= r["start"] <= r["end"] <= r["duration"] + 1e-6


def test_predict_never_returns_bounds_outside_the_recording(tmp_path):
    ckpt = _fresh_checkpoint(tmp_path)
    predict._CACHE.clear()
    x = np.random.default_rng(0).normal(0, 0.1, config.SR * 9).astype(np.float32)
    r = predict.predict_array(x, ckpt)
    if r["start"] is not None:
        assert r["start"] >= 0.0
        assert r["end"] <= r["duration"] + 1e-6


def test_cli_prints_valid_json(tmp_path):
    from tests.test_autolabel import _korotkoff
    from tools.make_annotator import wav_bytes

    ckpt = _fresh_checkpoint(tmp_path)
    wav = tmp_path / "s.wav"
    wav.write_bytes(wav_bytes(_korotkoff(dur=10.0) * 0.2, config.SR, bits=16))
    out = subprocess.run(
        [sys.executable, "-m", "src.predict", str(wav), "--ckpt", str(ckpt)],
        capture_output=True, text=True, cwd=str(config.PROJECT_ROOT),
    )
    assert out.returncode == 0, out.stderr
    parsed = json.loads(out.stdout)
    assert set(parsed) >= {"start", "end", "confidence", "duration"}
