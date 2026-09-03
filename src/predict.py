"""Инференс: путь к WAV -> границы интервала тонов Короткова.

    python -m src.predict "recording (100).wav"
    -> {"start": 6.12, "end": 17.44, "confidence": 0.91, "duration": 20.0}

Канал давления при инференсе не нужен: модель работает только по звуку.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import audio, config, model, postprocess

_CACHE = {}


def load_model(ckpt=None, device=None):
    ckpt = Path(ckpt or config.PROJECT_ROOT / "runs" / "best.pt")
    key = (str(ckpt), device)
    if key not in _CACHE:
        dev = model.pick_device(device)
        net = model.KorotkoffNet().to(dev)
        state = torch.load(ckpt, map_location=dev, weights_only=True)
        net.load_state_dict(state["state_dict"])
        net.eval()
        _CACHE[key] = (net, dev)
    return _CACHE[key]


def probabilities_from_array(x, ckpt=None):
    net, dev = load_model(ckpt)
    f = torch.from_numpy(audio.features(x))[None].to(dev)
    with torch.no_grad():
        return torch.sigmoid(net(f))[0].float().cpu().numpy()


def predict_array(x, ckpt=None, threshold=0.5, sr=config.SR):
    dur = len(x) / sr
    prob = probabilities_from_array(x, ckpt)
    r = postprocess.to_interval(prob, threshold=threshold)
    if r is None:
        return {"start": None, "end": None, "confidence": 0.0, "duration": dur}
    r["start"] = float(np.clip(r["start"], 0.0, dur))
    r["end"] = float(np.clip(r["end"], 0.0, dur))
    r["duration"] = dur
    return r


def predict(path, ckpt=None, threshold=0.5):
    x, _ = audio.load_8k(path)
    return predict_array(x, ckpt, threshold)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Интервал тонов Короткова в WAV")
    ap.add_argument("wav")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()
    print(json.dumps(predict(a.wav, a.ckpt, a.threshold), ensure_ascii=False, indent=2))
