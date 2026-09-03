"""Черновые метки интервала тонов Короткова.

ВАЖНО. Синхронизация аудио с каналом давления опровергнута (Task 5, отчёт
`reports/sync.md`), поэтому в конвейере работает только источник из звука.
Метки здесь — затравка для ручной проверки, а не эталон.

Функции `interval_from_pressure` и `combine` сохранены на случай, если
синхронизацию удастся установить другим способом, но `build` их не вызывает.

Способ построения повторяет осциллометрический: найти удары, взять их
амплитуды, сгладить в колокол, взять границы по долям максимума. Прежний
вариант с порогом по эвристическому score давал рваные интервалы.
"""
import csv

import numpy as np
from scipy import signal

from . import audio, config, manifest, pressure

# Доли максимума колокола, задающие границы. Подобраны всего на ЧЕТЫРЁХ
# записях (100, 599, 1000, 70) с границами, снятыми на глаз при разведке
# данных: средняя ошибка границы 0.67 с против 2.3 с при значении 0.35.
# Это заведомо слабая калибровка — пересчитать в Task 8 по калибровочному
# подмножеству размеченных человеком записей.
RISE = 0.15  # на подъёме -> начало
FALL = 0.15  # на спаде -> конец
MIN_BEATS = 6
BELL_SMOOTH_S = 2.5


def beat_bell(x, sr=config.SR):
    """Колоколообразный профиль громкости ударов на сетке кадров.

    Возвращает (bell, beat_frames) либо (None, None), если ударов слишком мало.
    """
    env = audio.rms_envelope(audio.bandpass(x, config.BAND_LO, config.BAND_HI, sr), sr)
    if len(env) < 20:
        return None, None
    bt = audio.beat_times(x, sr)
    if len(bt) < MIN_BEATS:
        return None, None

    frames = np.clip((bt / config.HOP_S).astype(int), 0, len(env) - 1)
    amps = env[frames].astype(np.float64)

    grid = np.arange(len(env))
    bell = np.interp(grid, frames, amps, left=amps[0], right=amps[-1])
    w = int(BELL_SMOOTH_S / config.HOP_S) | 1
    if len(bell) > w:
        bell = signal.savgol_filter(bell, w, 2)
    return np.clip(bell, 0.0, None), frames


def audio_confidence(x, sr=config.SR):
    """Насколько уверенно в записи вообще есть серия ударов, 0..1.

    Произведение двух вещей: выраженности колокола (пик против фона) и
    медианной периодичности внутри найденного интервала.
    """
    bell, frames = beat_bell(x, sr)
    if bell is None or bell.max() <= 0:
        return 0.0
    contrast = float(1.0 - np.percentile(bell, 10) / (bell.max() + 1e-12))
    env = audio.rms_envelope(audio.bandpass(x, config.BAND_LO, config.BAND_HI, sr), sr)
    per = audio.periodicity(env)
    imax = int(np.argmax(bell))
    lo = max(0, imax - int(3.0 / config.HOP_S))
    hi = min(len(per), imax + int(3.0 / config.HOP_S))
    local = float(np.median(per[lo:hi])) if hi > lo else 0.0
    return float(np.clip(contrast * local, 0.0, 1.0))


def interval_from_audio(x, sr=config.SR, rise=RISE, fall=FALL, min_conf=0.15):
    """Границы серии ударов по колоколу их амплитуд."""
    bell, _ = beat_bell(x, sr)
    if bell is None or bell.max() <= 0:
        return None
    if audio_confidence(x, sr) < min_conf:
        return None

    base = float(np.percentile(bell, 10))
    a = (bell - base) / (bell.max() - base + 1e-12)
    imax = int(np.argmax(a))

    up = np.where(a[:imax] < rise)[0]
    dn = np.where(a[imax:] < fall)[0]
    i0 = int(up[-1] + 1) if len(up) else 0
    i1 = int(imax + dn[0]) if len(dn) else len(a)

    dur = len(x) / sr
    start = float(np.clip(i0 * config.HOP_S, 0.0, dur))
    end = float(np.clip(i1 * config.HOP_S, 0.0, dur))
    if end - start < 0.5:
        return None
    return start, end


def interval_from_pressure(osc, rise=0.5, fall=0.7):
    """Границы по осциллометрической огибающей давления.

    В конвейере не используется: сопоставить эти границы с временной осью
    аудио нельзя, см. reports/sync.md.
    """
    env, t = osc["envelope"], osc["t"]
    if env.max() <= 0:
        return None
    a = env / env.max()
    imax = int(np.argmax(a))
    up = np.where(a[:imax] >= rise)[0]
    dn = np.where(a[imax:] <= fall)[0]
    if len(up) == 0 or len(dn) == 0:
        return None
    return float(t[up[0]]), float(t[imax + dn[0]])


def combine(p_iv, a_iv, offset):
    """Сводит два источника. p_iv задан в осях давления, a_iv — в осях аудио."""
    p_audio = None if p_iv is None else (p_iv[0] - offset, p_iv[1] - offset)

    if p_audio is None and a_iv is None:
        return None
    if p_audio is None:
        return {"start": a_iv[0], "end": a_iv[1], "disagreement": float("nan"),
                "priority": 2.0, "source": "audio"}
    if a_iv is None:
        return {"start": p_audio[0], "end": p_audio[1], "disagreement": float("nan"),
                "priority": 2.0, "source": "pressure"}

    dis = max(abs(p_audio[0] - a_iv[0]), abs(p_audio[1] - a_iv[1]))
    return {"start": float(0.5 * (p_audio[0] + a_iv[0])),
            "end": float(0.5 * (p_audio[1] + a_iv[1])),
            "disagreement": float(dis), "priority": float(dis / 1.5), "source": "both"}


FIELDS = ["id", "start", "end", "confidence", "priority", "source", "note"]


def build(rows=None):
    """Черновые метки по звуку для всех годных записей.

    priority — очередь ручной проверки: чем ниже уверенность и чем страннее
    длина интервала, тем раньше запись покажут человеку.
    """
    rows = rows or [r for r in manifest.read() if r["status"] in ("ok", "suspect")]
    out = []
    for r in rows:
        try:
            x, _ = audio.load_8k(r["wav_path"])
        except Exception:
            continue
        dur = len(x) / config.SR
        conf = audio_confidence(x)
        iv = interval_from_audio(x)
        notes = []

        if iv is None:
            start, end = 0.25 * dur, 0.75 * dur  # заведомо грубая заглушка
            notes.append("детектор не нашёл серию ударов")
            priority = 10.0
        else:
            start, end = iv
            priority = 1.0 - conf
            if end - start < 3.0:
                notes.append("интервал короче 3 с")
                priority += 3.0
            if end - start > 0.85 * dur:
                notes.append("интервал занимает почти всю запись")
                priority += 2.0
            if start < 0.5:
                notes.append("начало у самого края")
                priority += 1.0
            if end > dur - 0.5:
                notes.append("конец у самого края")
                priority += 1.0

        out.append({
            "id": r["id"], "start": round(start, 3), "end": round(end, 3),
            "confidence": round(conf, 4), "priority": round(priority, 4),
            "source": "audio", "note": "; ".join(notes),
        })
    return out


def write(rows, path=None):
    path = path or config.DATA_DIR / "labels_auto.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


if __name__ == "__main__":
    rows = build()
    write(rows)
    conf = np.array([r["confidence"] for r in rows])
    length = np.array([r["end"] - r["start"] for r in rows])
    print(f"размечено: {len(rows)}")
    print(f"уверенность: медиана {np.median(conf):.3f}, "
          f"p10 {np.percentile(conf, 10):.3f}, p90 {np.percentile(conf, 90):.3f}")
    print(f"длина интервала: медиана {np.median(length):.1f} с, "
          f"p10 {np.percentile(length, 10):.1f}, p90 {np.percentile(length, 90):.1f}")
    flagged = [r for r in rows if r["note"]]
    print(f"с замечаниями: {len(flagged)} ({100 * len(flagged) / max(len(rows), 1):.1f}%)")
    from collections import Counter
    for note, k in Counter(n for r in flagged for n in r["note"].split("; ")).most_common():
        print(f"    {note}: {k}")
