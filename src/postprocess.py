"""Кривая покадровых вероятностей -> две границы интервала."""
import numpy as np
from scipy import signal

from . import config


def smooth(prob, win_s=0.5):
    k = int(win_s / config.HOP_S) | 1
    p = np.asarray(prob, dtype=np.float64)
    if len(p) < k:
        return p.astype(np.float32)
    return signal.medfilt(p, kernel_size=k).astype(np.float32)


def close_gaps(mask, max_gap_s=3.0):
    """Сшивает короткие разрывы.

    Аускультативный провал — не конец интервала: у части гипертоников тоны
    пропадают в середине и возвращаются. Интервал задаётся внешними границами.
    """
    m = np.asarray(mask, dtype=bool).copy()
    max_gap = int(max_gap_s / config.HOP_S)
    idx = np.where(m)[0]
    if len(idx) < 2:
        return m
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_gap + 1:
            m[a:b] = True
    return m


def longest_run(mask):
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return None
    padded = np.concatenate([[False], m, [False]])
    d = np.diff(padded.astype(np.int8))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    k = int(np.argmax(ends - starts))
    return int(starts[k]), int(ends[k])


def refine_edge(prob, idx, threshold):
    """Уточняет границу линейной интерполяцией между кадрами idx-1 и idx.

    Формула одна для обоих концов: на подъёме p[idx-1] <= thr < p[idx],
    на спаде p[idx-1] > thr >= p[idx]. Числитель и знаменатель меняют знак
    вместе, поэтому доля выходит положительной в обоих случаях.
    """
    p = np.asarray(prob, dtype=np.float64)
    i = int(np.clip(idx, 1, len(p) - 1))
    a, b = p[i - 1], p[i]
    if abs(b - a) < 1e-9:
        return i * config.HOP_S
    frac = float(np.clip((threshold - a) / (b - a), 0.0, 1.0))
    return (i - 1 + frac) * config.HOP_S


def to_interval(prob, threshold=0.5, max_gap_s=3.0):
    p = smooth(prob)
    run = longest_run(close_gaps(p > threshold, max_gap_s))
    if run is None:
        return None
    a, b = run
    start = refine_edge(p, a, threshold) if a > 0 else 0.0
    end = refine_edge(p, b, threshold) if b < len(p) else len(p) * config.HOP_S
    if end <= start:
        end = start + config.HOP_S
    return {"start": float(start), "end": float(end),
            "confidence": float(np.mean(p[a:b])) if b > a else 0.0}
