import pytest

from src import config, manifest


def test_recording_id_extracts_number():
    assert manifest.recording_id("/x/recording (123).wav") == 123
    assert manifest.recording_id("recording (7).wav") == 7


def test_recording_id_rejects_unexpected_name():
    with pytest.raises(ValueError):
        manifest.recording_id("REC0.WAV")


def test_write_then_read_roundtrip(tmp_path):
    rows = [
        {
            "id": 1, "wav_path": "/a.wav", "pressure_path": "", "dur_s": 23.5,
            "sr_in": 16000, "dup_even": 1.0, "dup_odd": 0.31, "peak": 588.0,
            "clip_frac": 0.0, "hf_frac": 0.0002, "has_pressure": 0,
            "press_dur_s": "", "deflation_rate": "", "status": "no_pressure",
        }
    ]
    f = tmp_path / "m.csv"
    manifest.write(rows, f)
    back = manifest.read(f)
    assert len(back) == 1
    assert back[0]["id"] == 1
    assert abs(back[0]["dur_s"] - 23.5) < 1e-9
    assert back[0]["status"] == "no_pressure"


@pytest.mark.data
def test_build_on_real_data_matches_known_counts():
    if not config.recording_dir().exists():
        pytest.skip("нет исходных данных")
    rows = manifest.build()
    assert len(rows) == 906
    by = lambda s: sum(1 for r in rows if r["status"] == s)
    assert by("unreadable") == 1  # recording (221).wav — 819200 байт нулей
    assert by("excluded") == 1  # recording (199).wav — 17.7% энергии выше 4 кГц

    # По именам пара находится у 784 записей, но две из них негодны:
    # у 87 спуск манжеты оборвался на 1.43 с, у 221 мёртвое аудио.
    assert sum(1 for r in rows if r["pressure_path"]) == 783
    assert sum(1 for r in rows if r["has_pressure"]) == 782

    ok = [r for r in rows if r["status"] == "ok"]
    assert len(ok) > 750


@pytest.mark.data
def test_finds_pressure_file_with_missing_space_in_name():
    """Единственное исключение из шаблона именования: pressure1054.csv."""
    if not config.pressure_dir().exists():
        pytest.skip("нет исходных данных")
    p = manifest.pressure_csv_for(1054)
    assert p is not None and p.name == "pressure1054.csv"
