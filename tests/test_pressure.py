import numpy as np
import pytest

from src import config, pressure


def _synth(dur=25.0, p0=160.0, p1=40.0, bpm=72, peak_t=12.0, fs=100.0):
    """Синтетическая кривая: линейный спуск плюс пульсации с колоколообразной огибающей."""
    t = np.arange(0, dur, 1 / fs)
    base = p0 + (p1 - p0) * t / dur
    amp = 3.0 * np.exp(-0.5 * ((t - peak_t) / 3.5) ** 2)
    puls = amp * np.sin(2 * np.pi * (bpm / 60.0) * t)
    return t, base + puls


def test_read_pressure_parses_headerless_csv(tmp_path):
    f = tmp_path / "pressure 1.csv"
    f.write_text("0.29, 173.93\n0.31, 165.72\n0.32, 165.59\n")
    t, p = pressure.read_pressure(f)
    assert np.allclose(t, [0.29, 0.31, 0.32])
    assert np.allclose(p, [173.93, 165.72, 165.59])


def test_resample_uniform_gives_constant_step():
    t, p = _synth()
    tu, pu = pressure.resample_uniform(t, p)
    assert np.allclose(np.diff(tu), 1 / config.PRESSURE_FS)
    assert len(tu) == len(pu)


def test_oscillometric_finds_envelope_peak_near_truth():
    t, p = _synth(peak_t=12.0)
    tu, pu = pressure.resample_uniform(t, p)
    r = pressure.oscillometric(tu, pu)
    assert abs(r["map_t"] - 12.0) < 1.5


def test_oscillometric_ignores_valve_transient():
    """Ступень в первую секунду не должна становиться максимумом огибающей."""
    t, p = _synth(peak_t=15.0)
    p = p.copy()
    p[: int(0.4 * 100)] += 40.0  # резкий сброс клапана
    tu, pu = pressure.resample_uniform(t, p)
    r = pressure.oscillometric(tu, pu)
    assert r["map_t"] > config.PRESSURE_SKIP_S + 1.0
    assert abs(r["map_t"] - 15.0) < 2.5


def test_deflation_rate_is_negative_and_plausible():
    t, p = _synth(dur=25.0, p0=160.0, p1=40.0)
    tu, pu = pressure.resample_uniform(t, p)
    r = pressure.oscillometric(tu, pu)
    assert -8.0 < r["deflation_rate"] < -1.0


def test_pressure_at_interpolates():
    t, p = _synth()
    tu, pu = pressure.resample_uniform(t, p)
    v = pressure.pressure_at(tu, pu, 12.0)
    assert 40.0 < v < 160.0


@pytest.mark.data
def test_real_file_has_expected_shape():
    f = config.pressure_dir() / "pressure 100.csv"
    if not f.exists():
        pytest.skip("нет исходных данных")
    t, p = pressure.read_pressure(f)
    assert p[0] > 120 and abs(p[-1] - 40.0) < 2.0
    tu, pu = pressure.resample_uniform(t, p)
    r = pressure.oscillometric(tu, pu)
    assert -6.0 < r["deflation_rate"] < -2.0
    assert len(r["beat_times"]) > 10


def test_read_pressure_tolerates_truncated_last_row(tmp_path):
    """У pressure 145/267/275 запись оборвалась на середине последней строки."""
    f = tmp_path / "p.csv"
    f.write_text("0.10, 160.0\n0.12, 159.8\n0.14, 159.5\n0.16,")
    t, p = pressure.read_pressure(f)
    assert len(t) == 3
    assert abs(p[-1] - 159.5) < 1e-9


def test_read_pressure_tolerates_empty_second_field(tmp_path):
    f = tmp_path / "p2.csv"
    f.write_text("0.10, 160.0\n0.12, 159.8\n0.14, \n")
    t, p = pressure.read_pressure(f)
    assert len(t) == 2


def test_read_pressure_rejects_file_with_too_few_rows(tmp_path):
    f = tmp_path / "p3.csv"
    f.write_text("0.10, 160.0\n")
    with pytest.raises(ValueError, match="годных"):
        pressure.read_pressure(f)


def test_oscillometric_flags_recording_shorter_than_transient():
    """pressure 87.csv обрывается на 1.43 с — анализировать нечего."""
    t = np.arange(0, 1.4, 1 / config.PRESSURE_FS)
    p = 160 - 80 * t
    r = pressure.oscillometric(t, p)
    assert r["too_short"] is True
    assert len(r["beat_times"]) == 0
    assert np.isnan(r["deflation_rate"])


def test_oscillometric_does_not_flag_normal_recording():
    t, p = _synth()
    tu, pu = pressure.resample_uniform(t, p)
    assert pressure.oscillometric(tu, pu)["too_short"] is False
