"""Разбиение выборки — только непрерывными блоками номеров.

Случайное разбиение запрещено: соседние номера файлов почти наверняка
принадлежат одному испытуемому, и случайное деление даёт утечку между
обучением и тестом с завышенной оценкой качества.
"""
import csv
import json
from pathlib import Path

import numpy as np

from . import config


def usable_ids(ids):
    return [i for i in sorted(ids) if i not in config.EXCLUDED]


def block_split(ids, fractions=(0.7, 0.15, 0.15)):
    ids = sorted(ids)
    n = len(ids)
    n_tr = int(round(n * fractions[0]))
    n_va = int(round(n * fractions[1]))
    return {"train": ids[:n_tr], "val": ids[n_tr : n_tr + n_va], "test": ids[n_tr + n_va :]}


def split_verified(verified_ids, calib_frac=0.3):
    """Калибровочные записи настраивают пороги; тестовых не касается ничто."""
    ids = sorted(verified_ids)
    k = int(round(len(ids) * calib_frac))
    return ids[:k], ids[k:]


def read_verified(path=None):
    path = Path(path or config.DATA_DIR / "labels_verified.csv")
    out = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if str(row.get("verified", "1")).strip() in ("1", "true", "True"):
                out[int(row["id"])] = (float(row["start"]), float(row["end"]))
    return out


def calibrate_audio_ratios(calib_ids, verified=None):
    """Подбирает RISE/FALL звукового детектора по калибровочным записям.

    Раньше здесь калибровались коэффициенты осциллометрии, но после
    опровержения синхронизации (reports/sync.md) источником меток стал звук.
    """
    from . import audio, autolabel, manifest

    verified = verified if verified is not None else read_verified()
    rows = {r["id"]: r for r in manifest.read()}
    cache = {}
    for rid in calib_ids:
        r = rows.get(rid)
        if not r or rid not in verified:
            continue
        try:
            cache[rid] = audio.load_8k(r["wav_path"])[0]
        except Exception:
            continue
    if not cache:
        return (autolabel.RISE, autolabel.FALL), float("nan")

    best, best_err = (autolabel.RISE, autolabel.FALL), float("inf")
    for rise in np.arange(0.05, 0.60, 0.05):
        for fall in np.arange(0.05, 0.60, 0.05):
            errs = []
            for rid, x in cache.items():
                iv = autolabel.interval_from_audio(x, rise=float(rise), fall=float(fall))
                gs, ge = verified[rid]
                if iv is None:
                    errs += [9.9, 9.9]
                else:
                    errs += [abs(iv[0] - gs), abs(iv[1] - ge)]
            m = float(np.median(errs))
            if m < best_err:
                best_err, best = m, (float(rise), float(fall))
    return best, best_err


def save(d, path=None):
    path = Path(path or config.DATA_DIR / "splits.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(d, fh, indent=2)


def load(path=None):
    path = Path(path or config.DATA_DIR / "splits.json")
    with path.open() as fh:
        return json.load(fh)


if __name__ == "__main__":
    from . import manifest

    vpath = config.DATA_DIR / "labels_verified.csv"
    verified = read_verified(vpath) if vpath.exists() else {}
    calib, test_v = split_verified(list(verified))

    pool = usable_ids(
        r["id"] for r in manifest.read()
        if r["status"] in ("ok", "suspect") and r["id"] not in test_v
    )
    d = block_split(pool)
    d["verified_calib"] = calib
    d["verified_test"] = test_v
    save(d)

    print(f"train/val/test: {len(d['train'])}/{len(d['val'])}/{len(d['test'])}")
    if verified:
        print(f"проверено вручную: {len(verified)} "
              f"(калибровка {len(calib)}, тест {len(test_v)})")
        (rise, fall), err = calibrate_audio_ratios(calib, verified)
        print(f"калиброванные пороги: RISE={rise:.2f} FALL={fall:.2f}, "
              f"медианная ошибка границы {err:.2f} с")
        print("Внести в src/autolabel.py и перезапустить python -m src.autolabel")
    else:
        print(f"{vpath} пока нет — обучение пойдёт на черновых метках, "
              "итоговая оценка будет невозможна")
