import json

import numpy as np

from tools import make_annotator as mk


def test_wav_bytes_has_correct_header_and_size():
    x = (0.5 * np.sin(2 * np.pi * 50 * np.arange(4000) / 4000)).astype(np.float32)
    b = mk.wav_bytes(x, 4000)
    assert b[:4] == b"RIFF"
    assert b[8:12] == b"WAVE"
    assert len(b) == 44 + len(x) * 2


def test_wav_bytes_8bit_is_unsigned_and_half_the_size():
    """8 бит в WAV — беззнаковые со смещением 128, иначе получится треск."""
    x = np.array([0.0, 1.0, -1.0], dtype=np.float32)
    b = mk.wav_bytes(x, 4000, bits=8)
    assert len(b) == 44 + 3
    pcm = np.frombuffer(b[44:], dtype=np.uint8)
    assert pcm.tolist() == [128, 255, 1]
    assert int.from_bytes(b[34:36], "little") == 8


def test_playback_rate_is_decodable_by_browsers():
    """Ниже 3000 Гц браузер отказывается декодировать WAV — проверено в Chrome."""
    assert mk.PLAY_SR >= 3000


def test_wav_bytes_rejects_unsupported_depth():
    import pytest as _pytest

    with _pytest.raises(ValueError):
        mk.wav_bytes(np.zeros(4, dtype=np.float32), 4000, bits=24)


def test_wav_bytes_clips_instead_of_wrapping():
    """Переполнение int16 обернулось бы в громкий треск."""
    x = np.array([5.0, -5.0], dtype=np.float32)
    b = mk.wav_bytes(x, 4000)
    pcm = np.frombuffer(b[44:], dtype="<i2")
    assert pcm[0] == 32767
    assert pcm[1] == -32768


def test_build_payload_sorts_by_priority_descending():
    rows = [
        {"id": 1, "wav_path": "a", "dur_s": 10.0},
        {"id": 2, "wav_path": "b", "dur_s": 10.0},
        {"id": 3, "wav_path": "c", "dur_s": 10.0},
    ]
    labels = {
        1: {"start": 1.0, "end": 5.0, "priority": 0.2, "confidence": 0.9, "note": ""},
        2: {"start": 2.0, "end": 6.0, "priority": 9.9, "confidence": 0.1, "note": "плохо"},
        3: {"start": 3.0, "end": 7.0, "priority": 4.0, "confidence": 0.5, "note": ""},
    }
    p = mk.build_payload(rows, labels, limit=3, load_audio=False)
    assert [it["id"] for it in p["items"]] == [2, 3, 1]


def test_hard_strategy_slices_the_ranked_list_contiguously():
    rows = [{"id": i, "wav_path": str(i), "dur_s": 10.0} for i in range(1, 11)]
    labels = {i: {"start": 0.0, "end": 1.0, "priority": float(10 - i),
                  "confidence": 0.5, "note": ""} for i in range(1, 11)}
    first = mk.build_payload(rows, labels, limit=4, load_audio=False, strategy="hard")
    second = mk.build_payload(rows, labels, limit=4, offset=4, load_audio=False,
                              strategy="hard")
    assert [it["id"] for it in first["items"]] == [1, 2, 3, 4]
    assert [it["id"] for it in second["items"]] == [5, 6, 7, 8]


def test_offset_gives_a_different_batch_under_spread():
    """Второй файл разметки не должен повторять первый."""
    rows = [{"id": i, "wav_path": str(i), "dur_s": 10.0} for i in range(1, 41)]
    labels = {i: {"start": 0.0, "end": 1.0, "priority": float(40 - i),
                  "confidence": 0.5, "note": ""} for i in range(1, 41)}
    a = {it["id"] for it in mk.build_payload(rows, labels, limit=10,
                                             load_audio=False)["items"]}
    b = {it["id"] for it in mk.build_payload(rows, labels, limit=10, offset=1,
                                             load_audio=False)["items"]}
    assert a and b
    assert not (a & b), "выборки со смежными offset пересекаться не должны"


def test_build_payload_skips_records_without_labels():
    rows = [{"id": 1, "wav_path": "a", "dur_s": 10.0},
            {"id": 2, "wav_path": "b", "dur_s": 10.0}]
    labels = {2: {"start": 0.0, "end": 1.0, "priority": 1.0, "confidence": 0.5, "note": ""}}
    p = mk.build_payload(rows, labels, limit=5, load_audio=False)
    assert [it["id"] for it in p["items"]] == [2]


def test_render_inlines_payload_and_leaves_no_placeholder(tmp_path):
    tpl = tmp_path / "t.html"
    tpl.write_text("<html><body><script>const DATA=__PAYLOAD__;</script></body></html>")
    html = mk.render({"items": [{"id": 7}]}, tpl)
    assert "__PAYLOAD__" not in html
    body = html.split("const DATA=")[1].split(";</script>")[0]
    assert json.loads(body)["items"][0]["id"] == 7


def test_render_escapes_closing_script_tag(tmp_path):
    """Строка </script> внутри JSON оборвала бы разметку страницы."""
    tpl = tmp_path / "t.html"
    tpl.write_text("<script>const DATA=__PAYLOAD__;</script>")
    html = mk.render({"items": [{"note": "a </script> b"}]}, tpl)
    assert "a </script> b" not in html
    assert html.count("</script>") == 1


def test_downsample_is_robust_to_a_single_outlier():
    """Один артефактный выброс не должен прижимать всю кривую к нулю."""
    v = np.full(500, 0.2, dtype=np.float64)
    v[250] = 50.0
    out = np.array(mk._downsample(v, 100))
    assert out.max() <= 1.0
    assert np.median(out) > 0.5, "полезная часть кривой должна оставаться видимой"


def test_downsample_length_matches_request():
    v = np.linspace(0, 1, 977)
    assert len(mk._downsample(v, 120)) == 120


def test_playable_keeps_quiet_beats_audible_next_to_a_loud_artifact():
    """Нормировка по пику хоронит удары под одним хлопком."""
    from src import config as _cfg

    sr = _cfg.SR
    t = np.arange(0, 10.0, 1 / sr)
    x = (0.02 * np.sin(2 * np.pi * 60 * t)).astype(np.float32)
    x[int(5 * sr) : int(5 * sr) + 200] += 3.0  # артефактный хлопок
    y = mk._playable(x)
    assert np.max(np.abs(y)) <= 0.95
    assert float(np.mean(np.abs(y))) > 0.3, "полезный сигнал должен остаться слышимым"


def _fake(n):
    rows = [{"id": i, "dur_s": 20.0, "wav_path": f"/nope/{i}.wav"} for i in range(n)]
    labels = {i: {"start": 1.0, "end": 5.0, "priority": float(i), "confidence": 0.5,
                  "note": ""} for i in range(n)}
    return rows, labels


def test_spread_selection_covers_the_whole_priority_range():
    """Отбор одних трудных случаев сместил бы обучающую выборку."""
    rows, labels = _fake(400)
    got = mk.select(rows, labels, limit=40, offset=0, strategy="spread")
    prios = [labels[r["id"]]["priority"] for r in got]
    assert len(got) == 40
    assert max(prios) > 380, "самые трудные должны попасть"
    assert min(prios) < 20, "самые простые тоже"


def test_hard_selection_takes_only_the_worst():
    rows, labels = _fake(400)
    got = mk.select(rows, labels, limit=40, offset=0, strategy="hard")
    prios = [labels[r["id"]]["priority"] for r in got]
    assert min(prios) >= 360, "режим hard берёт только верх списка"


def test_selection_returns_no_duplicates():
    rows, labels = _fake(400)
    for strategy in ("spread", "hard"):
        got = mk.select(rows, labels, 40, 0, strategy)
        assert len({r["id"] for r in got}) == len(got)


def test_selection_handles_limit_larger_than_pool():
    rows, labels = _fake(10)
    assert len(mk.select(rows, labels, 50, 0, "spread")) == 10


def test_payload_carries_priority_for_client_side_sorting():
    rows, labels = _fake(5)
    payload = mk.build_payload(rows, labels, limit=5, load_audio=False)
    assert all("prio" in it for it in payload["items"])
