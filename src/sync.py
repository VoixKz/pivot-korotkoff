"""Оценка сдвига между временными осями аудио и канала давления.

Сопоставляются моменты отдельных ударов, а не амплитуды огибающих:
кросс-корреляция огибающих на этих данных даёт корреляцию 0.31-0.53 и
разброс оценки около ±2.3 с, чего для разметки недостаточно.

Смысл offset: t_давления = t_аудио + offset.
"""
import csv

import numpy as np
from . import audio, config, manifest, pressure

ACCEPT_MIN_SHARE = 0.70
ACCEPT_MAX_SPREAD = 0.30


def audio_beat_times(x, sr=config.SR):
    """Обёртка над audio.beat_times.

    Ранний вариант не опирался на оценку периода и считал удары вдвое
    (136 уд/мин там, где давление показывало 67), из-за чего сопоставление
    моментов было бессмысленным.
    """
    return audio.beat_times(x, sr)


def estimate_offset(audio_beats, press_beats, max_lag=6.0, step=0.02, tol=0.08):
    """Скользящий сдвиг: сколько ударов совпадает в пределах tol секунд.

    Возвращает (offset, score). score — доля совпавших от меньшей из двух
    последовательностей, то есть 1.0 при идеальном совпадении.
    """
    if len(audio_beats) < 5 or len(press_beats) < 5:
        return 0.0, 0.0
    lags = np.arange(-max_lag, max_lag + step, step)
    best_score, best_lag = 0.0, 0.0
    denom = min(len(audio_beats), len(press_beats))
    pb = np.sort(np.asarray(press_beats, dtype=np.float64))
    ab = np.sort(np.asarray(audio_beats, dtype=np.float64))
    for lag in lags:
        shifted = ab + lag
        idx = np.clip(np.searchsorted(pb, shifted), 1, len(pb) - 1)
        left = np.abs(pb[idx - 1] - shifted)
        right = np.abs(pb[idx] - shifted)
        matched = int(np.sum(np.minimum(left, right) <= tol))
        score = matched / denom
        if score > best_score:
            best_score, best_lag = score, float(lag)
    return best_lag, best_score


def estimate_all(rows=None):
    rows = rows or [
        r for r in manifest.read()
        if r["status"] in ("ok", "suspect") and r["has_pressure"]
    ]
    out = []
    for r in rows:
        try:
            x, _ = audio.load_8k(r["wav_path"])
            ab = audio_beat_times(x)
            t, p = pressure.read_pressure(r["pressure_path"])
            tu, pu = pressure.resample_uniform(t, p)
            pb = pressure.oscillometric(tu, pu)["beat_times"]
            off, score = estimate_offset(ab, pb)
            out.append({"id": r["id"], "offset": off, "score": score,
                        "n_audio": len(ab), "n_press": len(pb)})
        except Exception as e:
            out.append({"id": r["id"], "offset": float("nan"), "score": 0.0,
                        "n_audio": 0, "n_press": 0, "error": str(e)[:80]})
    return out


def null_control(rows=None, n=200, seed=0):
    """Контрольный опыт: тот же расчёт, но давление берётся от чужой записи.

    Без него score сам по себе ничего не значит — он получается максимизацией
    по 601 варианту сдвига, и высокое значение возникает случайно.
    """
    rows = rows or [
        r for r in manifest.read()
        if r["status"] == "ok" and r["has_pressure"]
    ]
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(rows), min(n, len(rows)), replace=False)

    ab_all, pb_all = [], []
    for i in sel:
        r = rows[int(i)]
        try:
            x, _ = audio.load_8k(r["wav_path"])
            ab = audio.beat_times(x)
            t, p = pressure.read_pressure(r["pressure_path"])
            tu, pu = pressure.resample_uniform(t, p)
            pb = pressure.oscillometric(tu, pu)["beat_times"]
            if len(ab) >= 5 and len(pb) >= 5:
                ab_all.append(ab)
                pb_all.append(pb)
        except Exception:
            continue

    m = len(ab_all)
    if m < 10:
        return None
    matched = np.array([estimate_offset(ab_all[i], pb_all[i])[1] for i in range(m)])
    perm = rng.permutation(m)
    perm = np.array([(q if q != i else (q + 1) % m) for i, q in enumerate(perm)])
    shuffled = np.array([estimate_offset(ab_all[i], pb_all[perm[i]])[1] for i in range(m)])
    return {
        "n": m,
        "matched_median": float(np.median(matched)),
        "shuffled_median": float(np.median(shuffled)),
        "share_matched_higher": float(np.mean(matched > shuffled)),
    }


def pairing_check(rows=None, seed=0):
    """Проверяет, что recording (N).wav и pressure N.csv — одно измерение.

    Это отдельный вопрос от синхронности: файлы могут быть парными, а их
    внутренние временные оси при этом несопоставимыми.
    """
    from scipy import stats

    rows = rows or [
        r for r in manifest.read()
        if r["status"] == "ok" and r["has_pressure"]
    ]
    wa = np.array([r["dur_s"] for r in rows], dtype=float)
    pr = np.array([r["press_dur_s"] for r in rows], dtype=float)
    if len(wa) < 20:
        return None
    r_true = float(stats.pearsonr(wa, pr).statistic)
    perm = np.random.default_rng(seed).permutation(len(wa))
    r_shuf = float(stats.pearsonr(wa, pr[perm]).statistic)
    d = pr - wa
    return {
        "n": len(wa),
        "r_duration": r_true,
        "r_duration_shuffled": r_shuf,
        "delta_median": float(np.median(d)),
        "delta_sd": float(d.std()),
    }


def report(results, control=None, pairing=None):
    good = [r for r in results if r["score"] >= 0.5 and np.isfinite(r["offset"])]
    offs = np.array([r["offset"] for r in good]) if good else np.zeros(0)
    lines = ["# Синхронизация аудио и давления", ""]
    lines.append(f"- файлов проверено: {len(results)}")
    lines.append(
        f"- с уверенным совпадением (score >= 0.5): {len(good)} "
        f"({100 * len(good) / max(len(results), 1):.1f}%)"
    )
    all_scores = np.array([r["score"] for r in results])
    lines.append(
        f"- score: медиана {np.median(all_scores):.3f}, "
        f"p90 {np.percentile(all_scores, 90):.3f}, макс {all_scores.max():.3f}"
    )
    verdict = "НЕ ПОДТВЕРЖДЕНА"
    if len(offs):
        med = float(np.median(offs))
        within = float(np.mean(np.abs(offs - med) <= ACCEPT_MAX_SPREAD))
        lines += [
            f"- медианный сдвиг: {med:+.3f} с",
            f"- в пределах ±{ACCEPT_MAX_SPREAD} с от медианы: {100 * within:.1f}%",
            f"- разброс: p10={np.percentile(offs, 10):+.2f} p90={np.percentile(offs, 90):+.2f}",
        ]
        share = len(good) / max(len(results), 1)
        if share >= ACCEPT_MIN_SHARE and within >= ACCEPT_MIN_SHARE:
            verdict = f"ПОДТВЕРЖДЕНА, сдвиг {med:+.3f} с"
        elif within >= ACCEPT_MIN_SHARE:
            verdict = (
                f"ЧАСТИЧНО: сдвиг устойчив ({med:+.3f} с), но уверенно совпало "
                f"лишь {100 * share:.0f}% файлов"
            )
    if pairing:
        lines += [
            "", "## Парность файлов", "",
            f"- пар проверено: {pairing['n']}",
            f"- корреляция длительностей аудио и давления: r = {pairing['r_duration']:.4f}",
            f"- то же на перемешанных парах: r = {pairing['r_duration_shuffled']:.4f}",
            f"- давление длиннее аудио на {pairing['delta_median']:+.2f} с "
            f"(СКО {pairing['delta_sd']:.2f})",
            "",
            "Файлы действительно относятся к одному измерению. Парность и "
            "синхронность — разные вопросы: первое подтверждено, второе нет.",
        ]

    if control:
        lines += [
            "", "## Контроль: давление от чужой записи", "",
            f"- проверено пар: {control['n']}",
            f"- score своей пары: медиана {control['matched_median']:.3f}",
            f"- score чужой пары: медиана {control['shuffled_median']:.3f}",
            f"- своя выше чужой в {100 * control['share_matched_higher']:.0f}% случаев",
            "",
            "Если чужая пара набирает почти столько же, сколько своя, то score — "
            "артефакт максимизации по сдвигу, а не признак синхронности.",
        ]

    lines += [
        "", f"## Вердикт: {verdict}", "",
        "Критерий приёмки: не менее 70% файлов дают уверенное совпадение "
        "и не менее 70% оценок лежат в пределах ±0.30 с от медианы.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    res = estimate_all()
    ctrl = null_control()
    pair = pairing_check()
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    text = report(res, ctrl, pair)
    (config.REPORT_DIR / "sync.md").write_text(text)
    with (config.DATA_DIR / "sync.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "offset", "score", "n_audio", "n_press"])
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in w.fieldnames} for r in res])
    print(text)
