"""Оценка на проверенном вручную тестовом наборе.

Метрики только временные. Пересчёт границ в мм рт. ст. и проверка по AAMI
отменены: они требуют синхронизации аудио с каналом давления, которая
опровергнута в Task 5 (см. reports/sync.md).
"""
import numpy as np

from . import config, manifest


def boundary_errors(pred, truth):
    ps, pe = pred
    ts, te = truth
    inter = max(0.0, min(pe, te) - max(ps, ts))
    union = max(pe, te) - min(ps, ts)
    return {"start_err": abs(ps - ts), "end_err": abs(pe - te),
            "iou": float(inter / union) if union > 0 else 0.0}


def summarise(results):
    if not results:
        return {"n": 0}
    se = np.array([r["start_err"] for r in results])
    ee = np.array([r["end_err"] for r in results])
    iou = np.array([r["iou"] for r in results])
    return {
        "n": len(results),
        "start_mae": float(se.mean()), "start_median": float(np.median(se)),
        "end_mae": float(ee.mean()), "end_median": float(np.median(ee)),
        "median_iou": float(np.median(iou)),
        "start_within_0.5": float(np.mean(se <= 0.5)),
        "start_within_1.0": float(np.mean(se <= 1.0)),
        "end_within_0.5": float(np.mean(ee <= 0.5)),
        "end_within_1.0": float(np.mean(ee <= 1.0)),
    }


def evaluate_on(ids, truth, predictor):
    rows = {r["id"]: r for r in manifest.read()}
    out, missed = [], 0
    for rid in ids:
        r = rows.get(rid)
        if not r or rid not in truth:
            continue
        try:
            p = predictor(r["wav_path"])
        except Exception:
            missed += 1
            continue
        if p.get("start") is None:
            missed += 1
            continue
        e = boundary_errors((p["start"], p["end"]), truth[rid])
        e["id"] = rid
        out.append(e)
    return out, missed


def estimate_subject_blocks(threshold=2.5):
    """Грубая оценка числа сессий по разрывам в ряду характеристик.

    Если записей много, а людей мало, эффективный размер выборки меньше
    числа файлов и любая оценка качества завышена. Это оценка порядка
    величины, а не точное число испытуемых.
    """
    rows = sorted(
        (r for r in manifest.read()
         if r["status"] == "ok" and r["deflation_rate"] != ""),
        key=lambda r: r["id"],
    )
    if len(rows) < 10:
        return {"n_recordings": len(rows), "n_blocks_estimate": len(rows),
                "recordings_per_block": 1.0}
    feats = np.array(
        [[r["dur_s"], np.log10(max(r["peak"], 1.0)), r["deflation_rate"]] for r in rows],
        dtype=float,
    )
    z = (feats - feats.mean(0)) / (feats.std(0) + 1e-9)
    jumps = np.linalg.norm(np.diff(z, axis=0), axis=1)
    n_blocks = int(np.sum(jumps > threshold)) + 1
    return {"n_recordings": len(rows), "n_blocks_estimate": n_blocks,
            "recordings_per_block": len(rows) / max(n_blocks, 1)}


def _fmt(s, missed=0):
    if s["n"] == 0:
        return "нет данных"
    lines = [
        f"- записей: {s['n']}" + (f" (не дал ответа на {missed})" if missed else ""),
        f"- начало: MAE {s['start_mae']:.2f} с, медиана {s['start_median']:.2f} с",
        f"- конец:  MAE {s['end_mae']:.2f} с, медиана {s['end_median']:.2f} с",
        f"- в пределах ±0.5 с: начало {100 * s['start_within_0.5']:.0f}%, "
        f"конец {100 * s['end_within_0.5']:.0f}%",
        f"- в пределах ±1.0 с: начало {100 * s['start_within_1.0']:.0f}%, "
        f"конец {100 * s['end_within_1.0']:.0f}%",
        f"- медианный IoU: {s['median_iou']:.3f}",
    ]
    return "\n".join(lines)


def report(model_s, base_s, subjects=None, model_missed=0, base_missed=0, n_train=None):
    if model_s["n"] == 0 or base_s["n"] == 0:
        verdict = "нет данных для сравнения"
    elif model_s["start_mae"] + model_s["end_mae"] < base_s["start_mae"] + base_s["end_mae"]:
        verdict = "модель лучше классического детектора"
    else:
        verdict = "**классический детектор не хуже модели**"

    lines = [
        "# Оценка на проверенном вручную тестовом наборе", "",
        "Тестовые записи не участвовали ни в обучении, ни в подборе порога, "
        "ни в калибровке коэффициентов разметки.", "",
        "Метрики только временные: пересчёт в мм рт. ст. невозможен, потому что "
        "аудио и канал давления не синхронизуемы (см. `reports/sync.md`).", "",
    ]
    if n_train:
        lines += [f"Модель обучена на {n_train} записях с черновыми метками.", ""]
    lines += ["## Модель", "", _fmt(model_s, model_missed), "",
              "## Классический baseline", "", _fmt(base_s, base_missed), "",
              f"## Итог: {verdict}", ""]
    if subjects:
        lines += [
            "## Сколько независимых сессий в данных", "",
            f"- записей: {subjects['n_recordings']}",
            f"- оценка числа сессий: {subjects['n_blocks_estimate']}",
            f"- записей на сессию: {subjects['recordings_per_block']:.1f}", "",
            "Оценка грубая, по разрывам в характеристиках соседних по номеру "
            "записей. Если записей на сессию много, эффективный размер выборки "
            "меньше числа файлов и оценка качества завышена.", "",
        ]
    return "\n".join(lines)


if __name__ == "__main__":
    from . import baseline, predict, splits

    sp = splits.load()
    vpath = config.DATA_DIR / "labels_verified.csv"
    if not vpath.exists():
        raise SystemExit(
            f"нет {vpath}: сначала нужна ручная разметка, иначе оценивать не с чем"
        )
    truth = splits.read_verified(vpath)
    ids = [i for i in sp.get("verified_test", []) if i in truth]
    if not ids:
        raise SystemExit("тестовый набор пуст: перезапустите python -m src.splits")

    m_res, m_missed = evaluate_on(ids, truth, lambda p: predict.predict(p))
    b_res, b_missed = evaluate_on(ids, truth, baseline.predict)

    n_train = None
    ckpt = config.PROJECT_ROOT / "runs" / "best.pt"
    if ckpt.exists():
        import torch

        n_train = torch.load(ckpt, map_location="cpu", weights_only=True).get("n_train")

    text = report(summarise(m_res), summarise(b_res), estimate_subject_blocks(),
                  m_missed, b_missed, n_train)
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORT_DIR / "evaluation.md").write_text(text)
    print(text)
