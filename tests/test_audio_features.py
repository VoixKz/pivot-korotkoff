import numpy as np
from scipy import signal as sg

from src import audio, config


def _pulse_train(dur_s, bpm, sr=config.SR, noise=0.0, start_s=0.0, end_s=None, seed=0):
    """Синтетические 'удары': короткие затухающие всплески 60 Гц с заданным пульсом."""
    n = int(dur_s * sr)
    x = np.random.default_rng(seed).normal(0, noise, n).astype(np.float32)
    end_s = dur_s if end_s is None else end_s
    period = 60.0 / bpm
    t_beat = np.arange(start_s, end_s, period)
    k = np.arange(int(0.08 * sr))
    burst = (np.sin(2 * np.pi * 60 * k / sr) * np.exp(-k / (0.02 * sr))).astype(np.float32)
    for tb in t_beat:
        i = int(tb * sr)
        if i + len(burst) <= n:
            x[i : i + len(burst)] += burst
    return x


def test_n_frames_matches_hop():
    assert audio.n_frames(config.SR * 10) == int(10 / config.HOP_S)


def test_rms_envelope_shape_and_peaks_at_beats():
    x = _pulse_train(10.0, 60)
    env = audio.rms_envelope(audio.bandpass(x, config.BAND_LO, config.BAND_HI))
    assert env.shape == (audio.n_frames(len(x)),)
    peaks, _ = sg.find_peaks(env, distance=int(0.5 / config.HOP_S), prominence=env.max() * 0.3)
    assert 8 <= len(peaks) <= 12


def test_periodicity_is_high_on_beats_and_low_on_noise():
    beats = _pulse_train(12.0, 72, noise=0.01)
    noise = np.random.default_rng(1).normal(0, 0.3, config.SR * 12).astype(np.float32)
    p_beats = audio.periodicity(
        audio.rms_envelope(audio.bandpass(beats, config.BAND_LO, config.BAND_HI))
    )
    p_noise = audio.periodicity(
        audio.rms_envelope(audio.bandpass(noise, config.BAND_LO, config.BAND_HI))
    )
    assert np.median(p_beats) > 0.45
    assert np.median(p_noise) < 0.35
    assert np.median(p_beats) > np.median(p_noise) + 0.15


def test_features_shape_and_normalisation():
    x = _pulse_train(20.0, 70, noise=0.02)
    f = audio.features(x)
    assert f.shape == (audio.n_frames(len(x)), config.N_FEATURES)
    assert f.dtype == np.float32
    assert np.isfinite(f).all()
    assert abs(float(f[:, : config.N_BANDS].mean())) < 0.2
    assert 0.5 < float(f[:, : config.N_BANDS].std()) < 2.0


def test_features_invariant_to_recording_gain():
    """Записи различаются по уровню в шесть раз — признаки не должны от этого зависеть."""
    x = _pulse_train(15.0, 65, noise=0.02)
    a = audio.features(x)
    b = audio.features(x * 6.0)
    assert np.allclose(a, b, atol=0.05)
