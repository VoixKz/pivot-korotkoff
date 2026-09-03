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


# --------------------------------------------------------------------------
# Признаки кадров
# --------------------------------------------------------------------------

def n_frames(n_samples, sr=config.SR):
    return int(n_samples // int(config.HOP_S * sr))


def bandpass(x, lo, hi, sr=config.SR, order=4):
    hi = min(hi, sr * 0.45)
    sos = signal.butter(order, [lo, hi], btype="band", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, x).astype(np.float32)


def rms_envelope(y, sr=config.SR):
    hop = int(config.HOP_S * sr)
    n = len(y) // hop
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    frames = y[: n * hop].reshape(n, hop)
    env = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1)) + 1e-9
    if n > 5:
        env = signal.savgol_filter(env, 5, 2)
    return np.clip(env, 1e-9, None).astype(np.float32)


def periodicity(env):
    """Для каждого кадра — максимум нормированной автокорреляции огибающей
    в диапазоне лагов, соответствующих правдоподобному пульсу.

    Считается через FFT сразу по всем окнам: наивный цикл с np.correlate
    занимает около полусекунды на запись, что неприемлемо при аугментации.
    """
    W = int(config.PERIODICITY_WIN_S / config.HOP_S)
    lo = int(config.IBI_MIN_S / config.HOP_S)
    hi = int(config.IBI_MAX_S / config.HOP_S)
    n = len(env)
    if n < lo + 4:
        return np.zeros(n, dtype=np.float32)

    # Паддинг отражением, а не повтором края: mode="edge" создаёт константный
    # отрезок, у которого нормированная автокорреляция равна единице на всех
    # лагах. Это давало ложный всплеск периодичности в первые секунды записи —
    # ровно там, где модель ищет границу начала.
    e = env.astype(np.float64)
    pad = np.pad(e, (W // 2, W - W // 2), mode="reflect")
    win = np.lib.stride_tricks.sliding_window_view(pad, W)[:n]
    z = win - win.mean(axis=1, keepdims=True)

    nfft = 1 << (2 * W - 1).bit_length()
    spec = np.fft.rfft(z, n=nfft, axis=1)
    ac = np.fft.irfft(spec * np.conj(spec), n=nfft, axis=1)[:, :W]

    denom = ac[:, :1].copy()
    denom[denom <= 0] = 1.0
    ac = ac / denom

    j = min(hi, W)
    if j <= lo:
        return np.zeros(n, dtype=np.float32)
    out = np.clip(ac[:, lo:j].max(axis=1), 0.0, None)

    # Почти постоянное окно не несёт сведений о периодичности, какой бы
    # ни вышла нормированная автокорреляция.
    scale = float(np.median(np.abs(e - np.median(e)))) + 1e-12
    out[win.std(axis=1) < 0.05 * scale] = 0.0
    return out.astype(np.float32)


def _log_filterbank(n_fft, sr=config.SR, n_bands=config.N_BANDS):
    """Треугольные полосы, равномерные по логарифму частоты в диапазоне MEL_LO..MEL_HI.

    В этом диапазоне мел-шкала почти линейна, поэтому логарифмическая честнее.
    """
    edges = np.geomspace(config.MEL_LO, config.MEL_HI, n_bands + 2)
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    fb = np.zeros((n_bands, len(freqs)), dtype=np.float64)
    for b in range(n_bands):
        lo, mid, hi = edges[b], edges[b + 1], edges[b + 2]
        left = (freqs >= lo) & (freqs <= mid)
        right = (freqs > mid) & (freqs <= hi)
        fb[b, left] = (freqs[left] - lo) / max(mid - lo, 1e-9)
        fb[b, right] = (hi - freqs[right]) / max(hi - mid, 1e-9)
        s = fb[b].sum()
        if s > 0:
            fb[b] /= s
    return fb


_FB_CACHE = {}


def log_band_energies(x, sr=config.SR):
    hop = int(config.HOP_S * sr)
    win = int(config.WIN_S * sr)
    n = len(x) // hop
    if n == 0:
        return np.zeros((0, config.N_BANDS), dtype=np.float32)
    n_fft = 1 << (win - 1).bit_length()
    if n_fft not in _FB_CACHE:
        _FB_CACHE[n_fft] = _log_filterbank(n_fft, sr)
    fb = _FB_CACHE[n_fft]
    window = np.hanning(win)
    padded = np.pad(x.astype(np.float64), (win // 2, win), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(padded, win)[: n * hop : hop] * window
    spec = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)) ** 2
    return np.log(spec @ fb.T + 1e-10).astype(np.float32)


def features(x, sr=config.SR):
    """(T, 42): 40 логарифмических полос + RMS-огибающая + периодичность.

    Нормировка внутри записи — уровень между файлами отличается в разы.

    Усиление снимается до вычисления полос, а не после. Иначе пол 1e-10 под
    логарифмом оказывается на разной относительной высоте для тихой и громкой
    записи, и z-нормировка этого уже не исправляет.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x / (np.sqrt(np.mean(x**2)) + 1e-12)
    y = bandpass(x, config.BAND_LO, config.BAND_HI, sr)
    env = rms_envelope(y, sr)
    per = periodicity(env)
    bands = log_band_energies(x, sr)

    T = min(len(env), len(per), len(bands))
    bands, env, per = bands[:T], env[:T], per[:T]

    bands = (bands - bands.mean()) / (bands.std() + 1e-6)
    log_env = np.log(env + 1e-9)
    log_env = (log_env - log_env.mean()) / (log_env.std() + 1e-6)

    return np.concatenate([bands, log_env[:, None], per[:, None]], axis=1).astype(np.float32)
