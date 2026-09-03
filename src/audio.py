"""Чтение WAV, приведение к рабочей частоте и признаки кадров.

Единственное место, где живёт знание о формате записей.
"""
import struct
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy import signal

from . import config


def read_wav(path):
    """Читает моно/стерео PCM16 WAV. Доверяет фактическому размеру файла, а не заголовку.

    Рекордер всегда объявляет в заголовке 60 секунд и обрывает запись раньше,
    поэтому размер чанка data читать нельзя — только фактические байты.
    """
    raw = Path(path).read_bytes()
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError(f"не RIFF/WAVE файл: {path}")
    pos, fmt = 12, None
    while pos + 8 <= len(raw):
        cid, size = struct.unpack_from("<4sI", raw, pos)
        body = pos + 8
        if cid == b"fmt ":
            fmt = struct.unpack_from("<HHIIHH", raw, body)
        elif cid == b"data":
            if fmt is None:
                raise ValueError(f"чанк data идёт раньше fmt: {path}")
            _, channels, sr, _, _, bits = fmt
            if bits != 16:
                raise ValueError(f"поддерживается только 16 бит, получено {bits}: {path}")
            avail = len(raw) - body
            n_bytes = min(size, avail) & ~1
            x = np.frombuffer(raw, dtype="<i2", count=n_bytes // 2, offset=body)
            if channels > 1:
                x = x[: len(x) - len(x) % channels].reshape(-1, channels).mean(axis=1)
            info = {
                "declared_bytes": int(size),
                "actual_bytes": int(avail),
                "channels": int(channels),
                "bits": int(bits),
            }
            return int(sr), x.astype(np.float32), info
        pos = body + size + (size & 1)
    raise ValueError(f"чанк data не найден: {path}")


def duplication_ratio(x, phase=0):
    """Доля соседних пар с равными отсчётами. phase=1 сдвигает разбиение на пары."""
    y = x[phase:]
    n = (len(y) // 2) * 2
    if n == 0:
        return 0.0
    pairs = y[:n].reshape(-1, 2)
    return float(np.mean(pairs[:, 0] == pairs[:, 1]))


def load_8k(path):
    """Читает файл и приводит к config.SR с антиалиасным фильтром.

    Не использует x[::2]: у части записей пары отсчётов сдвинуты на единицу,
    а одна запись сделана на 19200 Гц. resample_poly корректен во всех случаях.
    """
    sr, x, info = read_wav(path)
    if len(x) == 0:
        raise ValueError(f"пустой файл: {path}")

    diag = {
        "sr_in": sr,
        "n_in": len(x),
        "dur_s": len(x) / sr,
        "dup_even": duplication_ratio(x, 0),
        "dup_odd": duplication_ratio(x, 1),
        "peak": float(np.max(np.abs(x))),
        "clip_frac": float(np.mean(np.abs(x) >= 32700)),
        **info,
    }

    if sr != config.SR:
        r = Fraction(config.SR, sr).limit_denominator(1000)
        x = signal.resample_poly(x, r.numerator, r.denominator).astype(np.float32)

    nyq = config.SR / 2
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / config.SR)
    total = spec.sum() + 1e-12
    diag["hf_frac"] = float(spec[freqs > nyq * 0.8].sum() / total)
    diag["n_out"] = len(x)
    return x, diag
