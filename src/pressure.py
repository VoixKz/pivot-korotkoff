"""Канал давления манжеты: чтение и осциллометрическая огибающая.

Две вещи, установленные на реальных данных и потому зашитые в код:

1. Первые PRESSURE_SKIP_S секунд отбрасываются. Сброс клапана после накачки
   даёт ступень, на которой полосовой фильтр звенит; без этого максимум
   огибающей всегда ложно попадает на первую секунду.
2. Огибающая строится по амплитуде каждой отдельной пульсации, а не скользящим
   окном фиксированной ширины — окно смазывает колокол и смещает максимум.
"""
from pathlib import Path

import numpy as np
from scipy import signal

from . import config


def read_pressure(path):
    """Читает CSV вида `время_с, давление`.

    Разбор построчный, а не через np.loadtxt: у части файлов запись оборвалась
    на середине последней строки (`pressure 145`, `267`, `275`), и loadtxt на
    них падает целиком вместо потери одной строки.
    """
    ts, ps = [], []
    for line in Path(path).read_text().splitlines():
        parts = line.split(",")
        if len(parts) < 2:
            continue
        try:
            t, p = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        ts.append(t)
        ps.append(p)
    if len(ts) < 2:
        raise ValueError(f"меньше двух годных строк: {path}")
    return np.asarray(ts, dtype=np.float64), np.asarray(ps, dtype=np.float64)


def resample_uniform(t, p, fs=config.PRESSURE_FS):
    tu = np.arange(t[0], t[-1], 1.0 / fs)
    return tu, np.interp(tu, t, p)


def oscillometric(t_u, p_u, fs=config.PRESSURE_FS):
    skip = int(config.PRESSURE_SKIP_S * fs)

    # Запись короче переходного процесса анализировать нечем. Такое встречается
    # (`pressure 87.csv` обрывается на 1.43 с) — возвращаем пустой результат,
    # чтобы вызывающий пометил файл, а не ловил исключение.
    if len(p_u) < skip + int(2.0 * fs):
        return {
            "t": t_u, "pressure": p_u, "envelope": np.zeros_like(t_u),
            "beat_times": np.zeros(0), "beat_amps": np.zeros(0),
            "map_t": float("nan"), "map_mmhg": float("nan"),
            "deflation_rate": float("nan"), "too_short": True,
        }

    sos = signal.butter(3, list(config.PRESSURE_BAND), btype="band", fs=fs, output="sos")
    osc = signal.sosfiltfilt(sos, p_u)
    osc[:skip] = 0.0

    dist = max(1, int(config.IBI_MIN_S * fs))
    peaks, _ = signal.find_peaks(osc, distance=dist)
    troughs, _ = signal.find_peaks(-osc, distance=dist)
    peaks = peaks[peaks >= skip]
    troughs = troughs[troughs >= skip]

    beat_t, beat_a = [], []
    for pk in peaks:
        before = troughs[troughs < pk]
        after = troughs[troughs > pk]
        if len(before) == 0 or len(after) == 0:
            continue
        amp = osc[pk] - 0.5 * (osc[before[-1]] + osc[after[0]])
        if amp > 0:
            beat_t.append(t_u[pk])
            beat_a.append(amp)
    beat_t = np.asarray(beat_t)
    beat_a = np.asarray(beat_a)

    env = np.zeros_like(t_u)
    if len(beat_t) >= 3:
        env = np.interp(t_u, beat_t, beat_a, left=0.0, right=0.0)
        w = int(2.0 * fs) | 1
        if len(env) > w:
            env = signal.savgol_filter(env, w, 2)
        env = np.clip(env, 0.0, None)
    env[:skip] = 0.0

    imax = int(np.argmax(env)) if env.max() > 0 else skip
    valid = slice(skip, max(skip + 2, len(p_u) - 2))
    rate = float(np.median(np.gradient(p_u, 1.0 / fs)[valid]))

    return {
        "t": t_u,
        "pressure": p_u,
        "envelope": env,
        "beat_times": beat_t,
        "beat_amps": beat_a,
        "map_t": float(t_u[imax]),
        "map_mmhg": float(p_u[imax]),
        "deflation_rate": rate,
        "too_short": False,
    }


def pressure_at(t_u, p_u, t):
    return float(np.interp(t, t_u, p_u))
