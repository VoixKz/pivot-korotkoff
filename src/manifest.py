"""Обход исходных папок, сопоставление аудио с давлением, сбор диагностики."""
import csv
import re
from pathlib import Path

from . import audio, config, pressure

FIELDS = [
    "id", "wav_path", "pressure_path", "dur_s", "sr_in", "dup_even", "dup_odd",
    "peak", "clip_frac", "hf_frac", "has_pressure", "press_dur_s",
    "deflation_rate", "status",
]

_NUM = re.compile(r"recording \((\d+)\)\.wav$", re.IGNORECASE)


def recording_id(path):
    m = _NUM.search(str(path))
    if not m:
        raise ValueError(f"неожиданное имя файла: {path}")
    return int(m.group(1))


def pressure_csv_for(rid):
    """Путь к файлу давления для записи rid, либо None.

    Почти все файлы названы `pressure 123.csv`, но `pressure1054.csv` — без
    пробела. Проверять обе формы, иначе одна пара теряется молча.
    """
    for name in (f"pressure {rid}.csv", f"pressure{rid}.csv"):
        p = config.pressure_dir() / name
        if p.exists():
            return p
    return None


def build():
    rows = []
    for wav in sorted(config.recording_dir().glob("*.wav")):
        rid = recording_id(wav)
        row = {f: "" for f in FIELDS}
        row.update(id=rid, wav_path=str(wav), has_pressure=0, status="ok")

        try:
            _, diag = audio.load_8k(wav)
        except Exception:
            row["status"] = "unreadable"
            rows.append(row)
            continue

        row.update(
            dur_s=round(diag["dur_s"], 3), sr_in=diag["sr_in"],
            dup_even=round(diag["dup_even"], 4), dup_odd=round(diag["dup_odd"], 4),
            peak=diag["peak"], clip_frac=round(diag["clip_frac"], 6),
            hf_frac=round(diag["hf_frac"], 6),
        )

        if rid in config.EXCLUDED:
            row["status"] = "excluded"

        pcsv = pressure_csv_for(rid)
        if pcsv is not None:
            try:
                t, p = pressure.read_pressure(pcsv)
                tu, pu = pressure.resample_uniform(t, p)
                r = pressure.oscillometric(tu, pu)
                if r["too_short"]:
                    row.update(pressure_path=str(pcsv), press_dur_s=round(float(t[-1]), 3))
                    row["status"] = "suspect"
                else:
                    row.update(
                        pressure_path=str(pcsv), has_pressure=1,
                        press_dur_s=round(float(t[-1]), 3),
                        deflation_rate=round(r["deflation_rate"], 3),
                    )
            except Exception:
                row["status"] = "suspect"
        elif row["status"] == "ok":
            row["status"] = "no_pressure"

        # клиппинг или неожиданно широкая полоса — повод посмотреть глазами
        if row["status"] == "ok" and (diag["clip_frac"] > 0.001 or diag["hf_frac"] > 0.05):
            row["status"] = "suspect"

        rows.append(row)
    return rows


def write(rows, path=None):
    path = Path(path or config.DATA_DIR / "manifest.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


_NUMERIC = {
    "id": int, "sr_in": int, "has_pressure": int, "dur_s": float,
    "dup_even": float, "dup_odd": float, "peak": float, "clip_frac": float,
    "hf_frac": float, "press_dur_s": float, "deflation_rate": float,
}


def read(path=None):
    path = Path(path or config.DATA_DIR / "manifest.csv")
    out = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            for k, fn in _NUMERIC.items():
                if row.get(k, "") != "":
                    row[k] = fn(row[k])
            out.append(row)
    return out


if __name__ == "__main__":
    from collections import Counter

    rows = build()
    write(rows)
    print(f"записей: {len(rows)}")
    for k, v in Counter(r["status"] for r in rows).most_common():
        print(f"  {k}: {v}")
    print(f"  с давлением: {sum(r['has_pressure'] for r in rows)}")
