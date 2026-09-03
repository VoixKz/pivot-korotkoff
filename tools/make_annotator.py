"""Генератор автономной HTML-страницы для ручной выверки меток.

Страница работает офлайн: аудио и профили вшиты в файл, ничего не грузится
из сети. Кривая давления НЕ показывается сознательно — она не синхронна с
аудио (см. reports/sync.md), и её показ вводил бы разметчика в заблуждение.
"""
import argparse
import base64
import csv
import json
import struct
import sys
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import audio, config, manifest  # noqa: E402

# 4000 Гц, 8 бит. Частоту ниже 3000 Гц браузер не декодирует вовсе (проверено:
# на 2000 Гц decodeAudioData даёт EncodingError), а 8 бит возвращают тот же
# объём файла, что был у отвергнутого варианта 2000 Гц / 16 бит.
PLAY_SR = 4000
PLAY_BITS = 8
DRAW_HZ = 50  # точек профиля в секунду для рисования


def wav_bytes(x, sr, bits=16):
    """WAV в память. x — float примерно в диапазоне -1..1.

    16 бит — знаковый little-endian, 8 бит — беззнаковый со смещением 128,
    как того требует формат.
    """
    x = np.asarray(x, dtype=np.float64)
    if bits == 16:
        pcm = np.clip(x * 32767.0, -32768, 32767).astype("<i2").tobytes()
        block = 2
    elif bits == 8:
        pcm = np.clip(np.round(x * 127.0) + 128, 0, 255).astype(np.uint8).tobytes()
        block = 1
    else:
        raise ValueError(f"поддерживаются 8 или 16 бит, получено {bits}")
    hdr = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE"
    hdr += b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * block, block, bits)
    hdr += b"data" + struct.pack("<I", len(pcm))
    return hdr + pcm


def _playable(x):
    """Слышимая версия записи: полоса тонов, понижение частоты, нормировка.

    Нормировка по 90-му перцентилю с мягким ограничением через tanh, а не по
    пику: при нормировке по пику один артефактный хлопок делает все удары
    неслышными (средняя амплитуда падает до 0.010 при пике 0.9). Перцентиль
    поднимает основную часть записи в слышимый диапазон, а tanh не даёт
    хлопку превратиться в треск.
    """
    y = audio.bandpass(x, config.BAND_LO, config.BAND_HI)
    y = signal.resample_poly(y, PLAY_SR, config.SR).astype(np.float64)
    scale = float(np.percentile(np.abs(y), 90))
    if scale <= 0:
        scale = float(np.max(np.abs(y))) or 1.0
    return np.tanh(y / scale).astype(np.float32) * 0.9


def _downsample(v, n_out, robust=True):
    """Профиль для рисования.

    Нормировка по 95-му перцентилю, а не по максимуму: единственный артефактный
    выброс в 10 раз выше ударов прижимает всю кривую к нулю, и разметчик просто
    не видит, что размечает.
    """
    if len(v) == 0:
        return []
    idx = np.linspace(0, len(v), n_out + 1).astype(int)
    out = np.array([float(v[idx[i] : max(idx[i + 1], idx[i] + 1)].max()) for i in range(n_out)])
    scale = float(np.percentile(out, 95)) if robust else float(out.max())
    if scale <= 0:
        scale = float(out.max()) or 1.0
    return [round(x, 4) for x in np.clip(out / scale, 0.0, 1.0)]


def build_payload(rows, labels, limit=200, offset=0, load_audio=True):
    """Собирает данные страницы. Порядок — по убыванию приоритета проверки."""
    known = [r for r in rows if r["id"] in labels]
    known.sort(key=lambda r: (-labels[r["id"]]["priority"], r["id"]))
    chosen = known[offset : offset + limit]

    items = []
    for r in chosen:
        lab = labels[r["id"]]
        item = {
            "id": r["id"],
            "dur": float(r["dur_s"]),
            "start": float(lab["start"]),
            "end": float(lab["end"]),
            "conf": float(lab["confidence"]),
            "note": lab.get("note", ""),
            "env": [],
            "per": [],
            "audio": "",
        }
        if load_audio:
            x, _ = audio.load_8k(r["wav_path"])
            item["dur"] = len(x) / config.SR
            n_out = max(1, int(item["dur"] * DRAW_HZ))
            env = audio.rms_envelope(audio.bandpass(x, config.BAND_LO, config.BAND_HI))
            item["env"] = _downsample(env, n_out)
            item["per"] = _downsample(audio.periodicity(env), n_out)
            item["audio"] = base64.b64encode(
                wav_bytes(_playable(x), PLAY_SR, PLAY_BITS)
            ).decode()
        items.append(item)

    return {"items": items, "play_sr": PLAY_SR, "play_bits": PLAY_BITS, "draw_hz": DRAW_HZ,
            "offset": offset, "total": len(known)}


def render(payload, template_path):
    body = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return Path(template_path).read_text().replace("__PAYLOAD__", body)


def read_labels(path):
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            out[int(row["id"])] = {
                "start": float(row["start"]), "end": float(row["end"]),
                "priority": float(row["priority"]),
                "confidence": float(row["confidence"]),
                "note": row.get("note", ""),
            }
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Собрать HTML-разметчик")
    ap.add_argument("--limit", type=int, default=200, help="записей в одном файле")
    ap.add_argument("--offset", type=int, default=0, help="пропустить N по приоритету")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = [r for r in manifest.read() if r["status"] in ("ok", "suspect")]
    labels = read_labels(config.DATA_DIR / "labels_auto.csv")
    payload = build_payload(rows, labels, limit=a.limit, offset=a.offset)
    out = Path(a.out or config.DATA_DIR / f"annotator_{a.offset:04d}.html")
    out.write_text(render(payload, Path(__file__).parent / "annotator_template.html"))
    size = out.stat().st_size / 1e6
    print(f"готово: {out}")
    print(f"  записей: {len(payload['items'])} из {payload['total']} "
          f"(пропущено {a.offset}), размер {size:.1f} МБ")
    print("  открыть в браузере, выверить метки, нажать «Сохранить CSV»")
