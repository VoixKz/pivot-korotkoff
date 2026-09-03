"""Dataset для покадровой сегментации: признаки, маски, аугментации."""
import csv
from pathlib import Path

import numpy as np
import torch
from scipy import signal
from torch.utils.data import Dataset

from . import audio, config, manifest


def interval_to_mask(start, end, n_frames):
    m = np.zeros(n_frames, dtype=np.float32)
    a = int(np.clip(round(start / config.HOP_S), 0, n_frames))
    b = int(np.clip(round(end / config.HOP_S), 0, n_frames))
    m[a:b] = 1.0
    return m


def mask_to_interval(mask, threshold=0.5):
    idx = np.where(np.asarray(mask) > threshold)[0]
    if len(idx) == 0:
        return None
    return float(idx[0] * config.HOP_S), float((idx[-1] + 1) * config.HOP_S)


def read_labels(path):
    out = {}
    with Path(path).open(newline="") as fh:
        for row in csv.DictReader(fh):
            out[int(row["id"])] = (float(row["start"]), float(row["end"]))
    return out


def augment_signal(x, start, end, shift=0.0, stretch=1.0, gain=1.0, noise=0.0, rng=None):
    """Аугментация сигнала вместе с меткой.

    Метка обязана ехать вместе с сигналом: растяжение умножает границы,
    сдвиг прибавляет. Расхождение здесь отравило бы всю обучающую выборку.
    """
    rng = rng or np.random.default_rng(0)
    y = np.asarray(x, dtype=np.float32) * float(gain)

    if noise > 0:
        y = y + rng.normal(0, noise, len(y)).astype(np.float32)

    if abs(stretch - 1.0) > 1e-6:
        up = int(round(1000 * stretch))
        y = signal.resample_poly(y, up, 1000).astype(np.float32)
        start, end = start * stretch, end * stretch

    if abs(shift) > 1e-6:
        n = int(round(abs(shift) * config.SR))
        if shift > 0:
            y = np.concatenate([np.zeros(n, dtype=np.float32), y])
        else:
            y = y[n:] if n < len(y) else y[:1]
        start, end = start + shift, end + shift

    dur = len(y) / config.SR
    return y, float(np.clip(start, 0.0, dur)), float(np.clip(end, 0.0, dur))


class KorotkoffDataset(Dataset):
    def __init__(self, ids, labels, augment=False, cache=True, seed=0):
        self.rows = {r["id"]: r for r in manifest.read()}
        self.ids = [i for i in ids if i in labels and i in self.rows]
        self.labels = labels
        self.augment = augment
        self.cache = cache
        self.rng = np.random.default_rng(seed)
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.ids)

    def _signal(self, rid):
        return audio.load_8k(self.rows[rid]["wav_path"])[0]

    def _cached_features(self, rid):
        cpath = config.CACHE_DIR / f"{rid}.npy"
        if self.cache and cpath.exists():
            return np.load(cpath)
        f = audio.features(self._signal(rid))
        if self.cache:
            np.save(cpath, f)
        return f

    def __getitem__(self, i):
        rid = self.ids[i]
        start, end = self.labels[rid]

        if not self.augment:
            f = self._cached_features(rid)
            y = interval_to_mask(start, end, len(f))
            return torch.from_numpy(f), torch.from_numpy(y), rid

        x = self._signal(rid)
        quiet = x[: int(0.5 * config.SR)]
        noise_lvl = 0.0
        if len(quiet) and self.rng.random() < 0.5:
            noise_lvl = float(np.std(quiet)) * float(self.rng.uniform(0.5, 2.0))
        x, start, end = augment_signal(
            x, start, end,
            shift=float(self.rng.uniform(-1.5, 1.5)) if self.rng.random() < 0.5 else 0.0,
            stretch=float(self.rng.uniform(0.9, 1.1)) if self.rng.random() < 0.5 else 1.0,
            gain=float(self.rng.uniform(0.5, 2.0)),
            noise=noise_lvl,
            rng=self.rng,
        )
        f = audio.features(x)
        y = interval_to_mask(start, end, len(f))
        return torch.from_numpy(f), torch.from_numpy(y), rid


def collate(batch):
    lens = torch.tensor([b[0].shape[0] for b in batch], dtype=torch.long)
    T = int(lens.max())
    x = torch.zeros(len(batch), T, config.N_FEATURES, dtype=torch.float32)
    y = torch.zeros(len(batch), T, dtype=torch.float32)
    for i, (f, m, _) in enumerate(batch):
        x[i, : f.shape[0]] = f
        y[i, : m.shape[0]] = m
    return x, y, lens
