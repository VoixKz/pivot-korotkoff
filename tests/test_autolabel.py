import numpy as np

from src import autolabel, config


def _korotkoff(dur=26.0, start=7.0, end=18.0, bpm=68, sr=config.SR, noise=0.02, seed=0):
    """Запись с колоколообразной серией ударов между start и end.

    Амплитуда нарастает и спадает, как у тонов Короткова при спуске манжеты.
    """
    rng = np.random.default_rng(seed)
    n = int(dur * sr)
    x = rng.normal(0, noise, n).astype(np.float32)
    k = np.arange(int(0.09 * sr))
    burst = (np.sin(2 * np.pi * 55 * k / sr) * np.exp(-k / (0.022 * sr))).astype(np.float32)
    mid, half = 0.5 * (start + end), 0.5 * (end - start)
    for tb in np.arange(start, end, 60.0 / bpm):
        a = np.cos(0.5 * np.pi * (tb - mid) / max(half, 1e-6)) ** 2
        i = int(tb * sr)
        if i + len(burst) <= n:
            x[i : i + len(burst)] += (0.15 + 0.85 * a) * burst
    return x


def test_interval_from_audio_finds_the_beat_window():
    x = _korotkoff(start=7.0, end=18.0)
    iv = autolabel.interval_from_audio(x)
    assert iv is not None
    s, e = iv
    assert abs(s - 7.0) < 2.0, f"начало {s:.2f} вместо 7.0"
    assert abs(e - 18.0) < 2.0, f"конец {e:.2f} вместо 18.0"


def test_interval_from_audio_tracks_a_shifted_window():
    x = _korotkoff(dur=30.0, start=12.0, end=22.0, seed=1)
    s, e = autolabel.interval_from_audio(x)
    assert abs(s - 12.0) < 2.0
    assert abs(e - 22.0) < 2.0


def test_interval_from_audio_returns_none_on_pure_noise():
    x = np.random.default_rng(2).normal(0, 0.1, config.SR * 20).astype(np.float32)
    assert autolabel.interval_from_audio(x) is None


def test_interval_never_extends_beyond_recording():
    x = _korotkoff(dur=15.0, start=1.0, end=14.0, seed=3)
    s, e = autolabel.interval_from_audio(x)
    assert s >= 0.0
    assert e <= 15.0


def test_combine_uses_audio_when_pressure_missing():
    r = autolabel.combine(None, (3.0, 11.0), offset=1.0)
    assert (r["start"], r["end"]) == (3.0, 11.0)
    assert r["source"] == "audio"


def test_combine_returns_none_when_both_missing():
    assert autolabel.combine(None, None, offset=0.0) is None


def test_combine_applies_offset_to_pressure_side():
    """t_давления = t_аудио + offset, значит метки давления сдвигаются назад."""
    r = autolabel.combine((6.0, 16.0), None, offset=2.0)
    assert abs(r["start"] - 4.0) < 1e-6
    assert abs(r["end"] - 14.0) < 1e-6
    assert r["source"] == "pressure"


def test_priority_grows_with_disagreement():
    close = autolabel.combine((5.0, 15.0), (5.2, 15.1), offset=0.0)
    far = autolabel.combine((5.0, 15.0), (9.0, 20.0), offset=0.0)
    assert far["priority"] > close["priority"]


def test_confidence_is_lower_on_noise_than_on_beats():
    """Приоритет ручной проверки опирается на уверенность — она должна различать."""
    beats = _korotkoff(seed=4)
    noise = np.random.default_rng(5).normal(0, 0.1, config.SR * 26).astype(np.float32)
    assert autolabel.audio_confidence(beats) > autolabel.audio_confidence(noise)


def test_thresholds_roundtrip_through_the_file(tmp_path):
    """Откалиброванные пороги должны подхватываться без правки кода."""
    f = tmp_path / "thresholds.json"
    autolabel.save_thresholds(0.27, 0.33, median_error=0.41, n_calib=60, path=f)
    assert autolabel.load_thresholds(f) == (0.27, 0.33)


def test_load_thresholds_falls_back_on_a_broken_file(tmp_path):
    f = tmp_path / "broken.json"
    f.write_text("{ это не json")
    assert autolabel.load_thresholds(f) == (autolabel.DEFAULT_RISE, autolabel.DEFAULT_FALL)


def test_load_thresholds_falls_back_when_file_is_absent(tmp_path):
    got = autolabel.load_thresholds(tmp_path / "nope.json")
    assert got == (autolabel.DEFAULT_RISE, autolabel.DEFAULT_FALL)
