import struct
from pathlib import Path

import numpy as np
import pytest

from src import audio, config


def _write_wav(path, sr, samples, declared_bytes=None):
    """Пишет PCM16 WAV; declared_bytes позволяет соврать в заголовке, как это делает рекордер."""
    data = np.asarray(samples, dtype="<i2").tobytes()
    declared = len(data) if declared_bytes is None else declared_bytes
    hdr = b"RIFF" + struct.pack("<I", 36 + declared) + b"WAVE"
    hdr += b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
    hdr += b"data" + struct.pack("<I", declared)
    Path(path).write_bytes(hdr + data)


def test_reads_plain_wav(tmp_path):
    f = tmp_path / "a.wav"
    _write_wav(f, 16000, [0, 100, -100, 200])
    sr, x, info = audio.read_wav(f)
    assert sr == 16000
    assert x.tolist() == [0.0, 100.0, -100.0, 200.0]
    assert info["actual_bytes"] == 8


def test_trusts_actual_bytes_over_declared_header(tmp_path):
    """Рекордер объявляет 60 с, а пишет меньше — читать надо то, что есть."""
    f = tmp_path / "b.wav"
    _write_wav(f, 16000, [1, 2, 3, 4], declared_bytes=1920000)
    sr, x, info = audio.read_wav(f)
    assert len(x) == 4
    assert info["declared_bytes"] == 1920000
    assert info["actual_bytes"] == 8


def test_rejects_file_without_riff_header(tmp_path):
    f = tmp_path / "zeros.wav"
    f.write_bytes(b"\x00" * 1024)
    with pytest.raises(ValueError, match="RIFF"):
        audio.read_wav(f)


def test_load_8k_resamples_and_reports_diagnostics(tmp_path):
    """Синус 50 Гц, записанный на 16 кГц с дублированием отсчётов, должен выжить в 8 кГц."""
    sr_in = 16000
    t = np.arange(0, 4.0, 1 / (sr_in // 2))
    mono = (8000 * np.sin(2 * np.pi * 50 * t)).astype(np.int16)
    doubled = np.repeat(mono, 2)
    f = tmp_path / "c.wav"
    _write_wav(f, sr_in, doubled)

    x, diag = audio.load_8k(f)

    assert diag["sr_in"] == 16000
    assert diag["dup_even"] > 0.99
    assert abs(len(x) / config.SR - 4.0) < 0.05
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1 / config.SR)
    assert abs(freqs[int(np.argmax(spec))] - 50.0) < 1.0


def test_load_8k_handles_odd_phase_duplication(tmp_path):
    """У части записей пары сдвинуты на отсчёт — это не должно ничего ломать."""
    sr_in = 16000
    t = np.arange(0, 3.0, 1 / (sr_in // 2))
    mono = (8000 * np.sin(2 * np.pi * 40 * t)).astype(np.int16)
    doubled = np.repeat(mono, 2)[1:]  # роняем первый отсчёт
    f = tmp_path / "d.wav"
    _write_wav(f, sr_in, doubled)

    x, diag = audio.load_8k(f)
    assert diag["dup_even"] < 0.6
    assert diag["dup_odd"] > 0.99
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1 / config.SR)
    assert abs(freqs[int(np.argmax(spec))] - 40.0) < 1.0


def test_load_8k_handles_nonstandard_sample_rate(tmp_path):
    """recording (310).wav записан на 19200 Гц."""
    sr_in = 19200
    t = np.arange(0, 2.0, 1 / sr_in)
    f = tmp_path / "e.wav"
    _write_wav(f, sr_in, (5000 * np.sin(2 * np.pi * 60 * t)).astype(np.int16))
    x, diag = audio.load_8k(f)
    assert diag["sr_in"] == 19200
    assert abs(len(x) / config.SR - 2.0) < 0.05
