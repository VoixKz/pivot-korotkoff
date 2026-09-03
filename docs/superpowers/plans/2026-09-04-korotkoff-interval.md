# Детекция интервала тонов Короткова — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Обучить модель, которая по одному WAV-файлу измерения давления выдаёт время начала и конца интервала слышимых тонов Короткова.

**Architecture:** Покадровая сегментация. Аудио приводится к 8 кГц, разбивается на кадры по 20 мс, из каждого извлекается 42 признака. CRNN (расширенные свёртки + BiGRU) выдаёт вероятность «здесь есть удары» для каждого кадра. Постобработка превращает кривую вероятностей в два числа. Метки для обучения генерируются автоматически из синхронного канала давления манжеты и выборочно выверяются человеком.

**Tech Stack:** Python 3.14, NumPy 2.4, SciPy 1.18, PyTorch 2.14 (устройство MPS), pytest. Без librosa и без pandas — всё нужное есть в scipy и стандартной библиотеке.

**Spec:** `docs/superpowers/specs/2026-09-04-korotkoff-interval-design.md`

## Global Constraints

- Частота обработки: **8000 Гц**. Любой входной файл приводится к ней через `scipy.signal.resample_poly`. Никогда не использовать `x[::2]`.
- Шаг кадра **0.020 с**, окно **0.050 с**. Эти числа живут в `src/config.py` как `HOP_S` и `WIN_S`, больше нигде не дублируются.
- Число признаков на кадр: **42** (40 полос + огибающая + периодичность).
- Полоса полезного сигнала: **20–200 Гц**. Полоса лог-полос признаков: **20–500 Гц**.
- Диапазон правдоподобного пульса: **0.33–1.6 с** между ударами (37–180 уд/мин).
- Первые **2.0 с** канала давления отбрасываются всегда (переходный процесс клапана).
- Разбиение на train/val/test — **только непрерывными блоками номеров файлов**, никогда случайно.
- Исключённые файлы: `recording (221).wav`, `recording (199).wav`. Причины в спеке.
- Корень данных задаётся через `PIVOT_DATA_ROOT`, по умолчанию `/Users/BexKex_1/Downloads/Blood pressure raw data 906`. В коде путь не хардкодить.
- Все скрипты запускаются как модули из корня проекта: `python -m src.<имя>`.
- Тесты не должны требовать доступа ко всем 906 файлам: тяжёлые помечаются `@pytest.mark.data` и пропускаются, если данных нет.

## File Structure

| Файл | Ответственность |
|---|---|
| `src/config.py` | Все константы конвейера. Единственный источник истины для чисел. |
| `src/audio.py` | Чтение WAV, приведение к 8 кГц, диагностика качества, признаки кадров. |
| `src/pressure.py` | Чтение CSV давления, осциллометрическая огибающая. |
| `src/manifest.py` | Обход папок, сопоставление пар, сбор диагностики в `data/manifest.csv`. |
| `src/sync.py` | Оценка сдвига между осями аудио и давления. |
| `src/autolabel.py` | Метки из давления и из звука, их согласование, приоритет проверки. |
| `src/baseline.py` | Классический детектор без обучения — точка сравнения. |
| `src/splits.py` | Блочное разбиение по номерам, разделение проверенных файлов. |
| `src/dataset.py` | `torch.utils.data.Dataset`, аугментации, паддинг батчей. |
| `src/model.py` | Архитектура CRNN. |
| `src/train.py` | Цикл обучения, ранняя остановка, сохранение весов. |
| `src/postprocess.py` | Кривая вероятностей → интервал. |
| `src/predict.py` | CLI: путь к WAV → JSON. |
| `src/evaluate.py` | Метрики в секундах и в мм рт. ст., сравнение с baseline. |
| `tools/make_annotator.py` | Генератор автономного HTML-разметчика. |

---

### Task 1: Каркас проекта, конфигурация, чтение WAV

**Files:**
- Create: `requirements.txt`, `.gitignore`, `pytest.ini`
- Create: `src/__init__.py`, `src/config.py`, `src/audio.py`
- Test: `tests/test_audio_read.py`

**Interfaces:**
- Consumes: ничего
- Produces:
  - `config.SR = 8000`, `config.HOP_S = 0.020`, `config.WIN_S = 0.050`, `config.N_BANDS = 40`, `config.N_FEATURES = 42`, `config.BAND_LO = 20.0`, `config.BAND_HI = 200.0`, `config.MEL_HI = 500.0`, `config.IBI_MIN_S = 0.33`, `config.IBI_MAX_S = 1.6`, `config.PRESSURE_SKIP_S = 2.0`, `config.EXCLUDED = {221, 199}`, `config.data_root() -> pathlib.Path`
  - `audio.read_wav(path: str|Path) -> tuple[int, np.ndarray, dict]` — возвращает `(sample_rate, x_float32, info)`; `info` содержит `declared_bytes`, `actual_bytes`, `channels`, `bits`
  - `audio.load_8k(path) -> tuple[np.ndarray, dict]` — сигнал на 8000 Гц плюс диагностика

- [ ] **Step 1: Инициализировать репозиторий и зависимости**

Проект сейчас не под контролем версий, а план опирается на коммиты после каждой задачи.

```bash
cd "/Users/BexKex_1/Documents/UNREAL ACTIVITY/PivotBME/DeepLearning"
git init
printf '%s\n' 'data/cache/' 'data/*.csv' '__pycache__/' '*.pyc' '.pytest_cache/' 'runs/' '*.pt' > .gitignore
printf '%s\n' 'numpy>=2.4' 'scipy>=1.18' 'torch>=2.14' 'pytest>=8' > requirements.txt
printf '%s\n' '[pytest]' 'markers =' '    data: требует доступа к исходным данным' > pytest.ini
python3 -m pip install -r requirements.txt
```

- [ ] **Step 2: Написать падающий тест на чтение WAV**

```python
# tests/test_audio_read.py
import numpy as np, struct, pytest
from pathlib import Path
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
```

- [ ] **Step 3: Убедиться, что тест падает**

Run: `python -m pytest tests/test_audio_read.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.audio'`

- [ ] **Step 4: Написать `src/config.py`**

```python
# src/config.py
import os
from pathlib import Path

SR = 8000
HOP_S = 0.020
WIN_S = 0.050
N_BANDS = 40
N_FEATURES = N_BANDS + 2

BAND_LO, BAND_HI = 20.0, 200.0
MEL_LO, MEL_HI = 20.0, 500.0

IBI_MIN_S, IBI_MAX_S = 0.33, 1.6
PERIODICITY_WIN_S = 4.0

PRESSURE_SKIP_S = 2.0
PRESSURE_FS = 50.0
PRESSURE_BAND = (0.7, 6.0)

SYNC_OFFSET_S = 0.0   # заполняется по результату Task 5: t_давления = t_аудио + offset

EXCLUDED = {221, 199}

DEFAULT_DATA_ROOT = "/Users/BexKex_1/Downloads/Blood pressure raw data 906"

def data_root() -> Path:
    return Path(os.environ.get("PIVOT_DATA_ROOT", DEFAULT_DATA_ROOT))

def recording_dir() -> Path:
    return data_root() / "recording"

def pressure_dir() -> Path:
    return data_root() / "pressure"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
REPORT_DIR = PROJECT_ROOT / "reports"
```

- [ ] **Step 5: Написать `read_wav` в `src/audio.py`**

Разбор чанков вручную, потому что заголовок врёт про размер данных, а `scipy.io.wavfile` на этом ругается предупреждением на каждый файл.

```python
# src/audio.py
import struct
from pathlib import Path
import numpy as np
from scipy import signal
from . import config

def read_wav(path):
    """Читает моно/стерео PCM16 WAV. Доверяет фактическому размеру файла, а не заголовку."""
    raw = Path(path).read_bytes()
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError(f"не RIFF/WAVE файл: {path}")
    pos, fmt = 12, None
    while pos + 8 <= len(raw):
        cid, size = struct.unpack_from("<4sI", raw, pos)
        body = pos + 8
        if cid == b"fmt ":
            fmt = struct.unpack_from("<HHIIHH", raw, body)
        elif cid == b"data":
            if fmt is None:
                raise ValueError(f"чанк data идёт раньше fmt: {path}")
            _, channels, sr, _, _, bits = fmt
            if bits != 16:
                raise ValueError(f"поддерживается только 16 бит, получено {bits}: {path}")
            avail = len(raw) - body
            n_bytes = min(size, avail) & ~1
            x = np.frombuffer(raw, dtype="<i2", count=n_bytes // 2, offset=body)
            if channels > 1:
                x = x[: len(x) - len(x) % channels].reshape(-1, channels).mean(axis=1)
            info = {"declared_bytes": int(size), "actual_bytes": int(avail),
                    "channels": int(channels), "bits": int(bits)}
            return int(sr), x.astype(np.float32), info
        pos = body + size + (size & 1)
    raise ValueError(f"чанк data не найден: {path}")
```

- [ ] **Step 6: Убедиться, что тесты проходят**

Run: `python -m pytest tests/test_audio_read.py -v`
Expected: PASS, 3 теста

- [ ] **Step 7: Написать падающий тест на приведение к 8 кГц**

```python
# добавить в tests/test_audio_read.py
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
    # энергия должна остаться на 50 Гц
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1 / config.SR)
    assert abs(freqs[int(np.argmax(spec))] - 50.0) < 1.0

def test_load_8k_handles_odd_phase_duplication(tmp_path):
    """У части записей пары сдвинуты на отсчёт — это не должно ничего ломать."""
    sr_in = 16000
    t = np.arange(0, 3.0, 1 / (sr_in // 2))
    mono = (8000 * np.sin(2 * np.pi * 40 * t)).astype(np.int16)
    doubled = np.repeat(mono, 2)[1:]        # роняем первый отсчёт
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
```

- [ ] **Step 8: Убедиться, что тесты падают**

Run: `python -m pytest tests/test_audio_read.py -v`
Expected: FAIL — `AttributeError: module 'src.audio' has no attribute 'load_8k'`

- [ ] **Step 9: Реализовать `load_8k`**

```python
# добавить в src/audio.py
from fractions import Fraction

def duplication_ratio(x, phase=0):
    """Доля соседних пар с равными отсчётами. phase=1 сдвигает разбиение на пары."""
    y = x[phase:]
    n = (len(y) // 2) * 2
    if n == 0:
        return 0.0
    pairs = y[:n].reshape(-1, 2)
    return float(np.mean(pairs[:, 0] == pairs[:, 1]))

def load_8k(path):
    """Читает файл и приводит к config.SR с антиалиасным фильтром.

    Не использует x[::2]: у части записей пары отсчётов сдвинуты на единицу,
    а одна запись сделана на 19200 Гц. resample_poly корректен во всех случаях.
    """
    sr, x, info = read_wav(path)
    if len(x) == 0:
        raise ValueError(f"пустой файл: {path}")

    diag = {
        "sr_in": sr,
        "n_in": len(x),
        "dur_s": len(x) / sr,
        "dup_even": duplication_ratio(x, 0),
        "dup_odd": duplication_ratio(x, 1),
        "peak": float(np.max(np.abs(x))),
        "clip_frac": float(np.mean(np.abs(x) >= 32700)),
        **info,
    }

    if sr != config.SR:
        r = Fraction(config.SR, sr).limit_denominator(1000)
        x = signal.resample_poly(x, r.numerator, r.denominator).astype(np.float32)

    nyq = config.SR / 2
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / config.SR)
    total = spec.sum() + 1e-12
    diag["hf_frac"] = float(spec[freqs > nyq * 0.8].sum() / total)
    diag["n_out"] = len(x)
    return x, diag
```

- [ ] **Step 10: Убедиться, что тесты проходят**

Run: `python -m pytest tests/test_audio_read.py -v`
Expected: PASS, 6 тестов

- [ ] **Step 11: Проверить на реальных данных**

```bash
python -c "
from src import audio, config
import glob
bad=[]
for f in sorted(glob.glob(str(config.recording_dir()/'*.wav'))):
    try: audio.load_8k(f)
    except Exception as e: bad.append((f.split('/')[-1], str(e)[:60]))
print('не прочитано:', len(bad))
for b in bad: print('  ', b)
"
```
Expected: ровно один файл — `recording (221).wav` (нули без заголовка). Если список длиннее — разбираться, не идти дальше.

- [ ] **Step 12: Коммит**

```bash
git add -A
git commit -m "feat: чтение WAV и приведение к 8 кГц с диагностикой качества"
```

---

### Task 2: Признаки кадров

**Files:**
- Modify: `src/audio.py` (дописать блок признаков)
- Test: `tests/test_audio_features.py`

**Interfaces:**
- Consumes: `audio.load_8k`, `config.*`
- Produces:
  - `audio.bandpass(x, lo, hi, sr=config.SR, order=4) -> np.ndarray`
  - `audio.rms_envelope(x, sr=config.SR) -> np.ndarray` — форма `(T,)`, шаг `HOP_S`
  - `audio.periodicity(env) -> np.ndarray` — форма `(T,)`, значения 0..1
  - `audio.log_band_energies(x) -> np.ndarray` — форма `(T, 40)`
  - `audio.features(x) -> np.ndarray` — форма `(T, 42)`, float32, нормировано по записи
  - `audio.n_frames(n_samples) -> int`

- [ ] **Step 1: Написать падающие тесты**

```python
# tests/test_audio_features.py
import numpy as np
from src import audio, config

def _pulse_train(dur_s, bpm, sr=config.SR, noise=0.0, start_s=0.0, end_s=None):
    """Синтетические 'удары': короткие затухающие всплески 60 Гц с заданным пульсом."""
    n = int(dur_s * sr)
    x = np.random.default_rng(0).normal(0, noise, n).astype(np.float32)
    end_s = dur_s if end_s is None else end_s
    period = 60.0 / bpm
    t_beat = np.arange(start_s, end_s, period)
    k = np.arange(int(0.08 * sr))
    burst = (np.sin(2 * np.pi * 60 * k / sr) * np.exp(-k / (0.02 * sr))).astype(np.float32)
    for tb in t_beat:
        i = int(tb * sr)
        if i + len(burst) <= n:
            x[i:i + len(burst)] += burst
    return x

def test_n_frames_matches_hop():
    assert audio.n_frames(config.SR * 10) == int(10 / config.HOP_S)

def test_rms_envelope_shape_and_peaks_at_beats():
    x = _pulse_train(10.0, 60)
    env = audio.rms_envelope(audio.bandpass(x, config.BAND_LO, config.BAND_HI))
    assert env.shape == (audio.n_frames(len(x)),)
    # при 60 уд/мин ожидаем около 10 пиков
    from scipy import signal as sg
    peaks, _ = sg.find_peaks(env, distance=int(0.5 / config.HOP_S), prominence=env.max() * 0.3)
    assert 8 <= len(peaks) <= 12

def test_periodicity_is_high_on_beats_and_low_on_noise():
    beats = _pulse_train(12.0, 72, noise=0.01)
    noise = np.random.default_rng(1).normal(0, 0.3, config.SR * 12).astype(np.float32)
    p_beats = audio.periodicity(audio.rms_envelope(audio.bandpass(beats, config.BAND_LO, config.BAND_HI)))
    p_noise = audio.periodicity(audio.rms_envelope(audio.bandpass(noise, config.BAND_LO, config.BAND_HI)))
    assert np.median(p_beats) > 0.45
    assert np.median(p_noise) < 0.35
    assert np.median(p_beats) > np.median(p_noise) + 0.15

def test_features_shape_and_normalisation():
    x = _pulse_train(20.0, 70, noise=0.02)
    f = audio.features(x)
    assert f.shape == (audio.n_frames(len(x)), config.N_FEATURES)
    assert f.dtype == np.float32
    assert np.isfinite(f).all()
    # первые 40 каналов нормированы по записи
    assert abs(float(f[:, :config.N_BANDS].mean())) < 0.2
    assert 0.5 < float(f[:, :config.N_BANDS].std()) < 2.0

def test_features_invariant_to_recording_gain():
    """Записи различаются по уровню в шесть раз — признаки не должны от этого зависеть."""
    x = _pulse_train(15.0, 65, noise=0.02)
    a = audio.features(x)
    b = audio.features(x * 6.0)
    assert np.allclose(a, b, atol=0.05)
```

- [ ] **Step 2: Убедиться, что тесты падают**

Run: `python -m pytest tests/test_audio_features.py -v`
Expected: FAIL — `AttributeError: module 'src.audio' has no attribute 'bandpass'`

- [ ] **Step 3: Реализовать признаки**

```python
# добавить в src/audio.py

def n_frames(n_samples, sr=config.SR):
    return int(n_samples // int(config.HOP_S * sr))

def bandpass(x, lo, hi, sr=config.SR, order=4):
    hi = min(hi, sr * 0.45)
    sos = signal.butter(order, [lo, hi], btype="band", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, x).astype(np.float32)

def rms_envelope(y, sr=config.SR):
    hop = int(config.HOP_S * sr)
    n = len(y) // hop
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    frames = y[: n * hop].reshape(n, hop)
    env = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1)) + 1e-9
    if n > 5:
        env = signal.savgol_filter(env, 5, 2)
    return np.clip(env, 1e-9, None).astype(np.float32)

def periodicity(env):
    """Для каждого кадра — максимум нормированной автокорреляции огибающей
    в диапазоне лагов, соответствующих правдоподобному пульсу."""
    W = int(config.PERIODICITY_WIN_S / config.HOP_S)
    lo = int(config.IBI_MIN_S / config.HOP_S)
    hi = int(config.IBI_MAX_S / config.HOP_S)
    n = len(env)
    out = np.zeros(n, dtype=np.float32)
    if n < lo + 4:
        return out
    pad = np.pad(env.astype(np.float64), (W // 2, W // 2), mode="edge")
    for i in range(n):
        seg = pad[i:i + W]
        z = seg - seg.mean()
        denom = float(np.dot(z, z))
        if denom <= 0:
            continue
        ac = np.correlate(z, z, "full")[len(z) - 1:] / denom
        j = min(hi, len(ac))
        if j > lo:
            out[i] = max(0.0, float(ac[lo:j].max()))
    return out

def _log_filterbank(n_fft, sr=config.SR, n_bands=config.N_BANDS):
    """Треугольные полосы, равномерные по логарифму частоты в диапазоне MEL_LO..MEL_HI.
    В этом диапазоне мел-шкала почти линейна, поэтому логарифмическая честнее."""
    edges = np.geomspace(config.MEL_LO, config.MEL_HI, n_bands + 2)
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    fb = np.zeros((n_bands, len(freqs)), dtype=np.float64)
    for b in range(n_bands):
        lo, mid, hi = edges[b], edges[b + 1], edges[b + 2]
        left = (freqs >= lo) & (freqs <= mid)
        right = (freqs > mid) & (freqs <= hi)
        fb[b, left] = (freqs[left] - lo) / max(mid - lo, 1e-9)
        fb[b, right] = (hi - freqs[right]) / max(hi - mid, 1e-9)
        s = fb[b].sum()
        if s > 0:
            fb[b] /= s
    return fb

_FB_CACHE = {}

def log_band_energies(x, sr=config.SR):
    hop = int(config.HOP_S * sr)
    win = int(config.WIN_S * sr)
    n = len(x) // hop
    if n == 0:
        return np.zeros((0, config.N_BANDS), dtype=np.float32)
    n_fft = 1 << (win - 1).bit_length()
    if n_fft not in _FB_CACHE:
        _FB_CACHE[n_fft] = _log_filterbank(n_fft, sr)
    fb = _FB_CACHE[n_fft]
    window = np.hanning(win)
    padded = np.pad(x.astype(np.float64), (win // 2, win), mode="reflect")
    idx = np.arange(n) * hop
    frames = np.stack([padded[i:i + win] * window for i in idx])
    spec = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)) ** 2
    return np.log(spec @ fb.T + 1e-10).astype(np.float32)

def features(x, sr=config.SR):
    """(T, 42): 40 логарифмических полос + RMS-огибающая + периодичность.
    Нормировка внутри записи — уровень между файлами отличается в разы."""
    y = bandpass(x, config.BAND_LO, config.BAND_HI, sr)
    env = rms_envelope(y, sr)
    per = periodicity(env)
    bands = log_band_energies(x, sr)

    T = min(len(env), len(per), len(bands))
    bands, env, per = bands[:T], env[:T], per[:T]

    bands = (bands - bands.mean()) / (bands.std() + 1e-6)
    log_env = np.log(env + 1e-9)
    log_env = (log_env - log_env.mean()) / (log_env.std() + 1e-6)

    return np.concatenate(
        [bands, log_env[:, None], per[:, None]], axis=1
    ).astype(np.float32)
```

- [ ] **Step 4: Убедиться, что тесты проходят**

Run: `python -m pytest tests/test_audio_features.py -v`
Expected: PASS, 5 тестов

- [ ] **Step 5: Проверить на реальном файле с известной картиной**

```bash
python -c "
from src import audio, config
x,_ = audio.load_8k(config.recording_dir()/'recording (100).wav')
f = audio.features(x)
print('shape', f.shape, 'ожидалось примерно', int(20.0/config.HOP_S), 'x 42')
per = f[:,41]
import numpy as np
t = np.arange(len(per))*config.HOP_S
hot = t[per > 0.5]
print('периодичность выше 0.5 на интервале %.1f..%.1f с' % (hot.min(), hot.max()))
"
```
Expected: форма примерно `(1000, 42)`; интервал высокой периодичности пересекается с 6–17 с — тем, что видно на огибающей этого файла.

- [ ] **Step 6: Коммит**

```bash
git add -A
git commit -m "feat: признаки кадров - логполосы, огибающая, периодичность"
```

---

### Task 3: Канал давления и осциллометрия

**Files:**
- Create: `src/pressure.py`
- Test: `tests/test_pressure.py`

**Interfaces:**
- Consumes: `config.*`
- Produces:
  - `pressure.read_pressure(path) -> tuple[np.ndarray, np.ndarray]` — `(t, p)` как есть из файла
  - `pressure.resample_uniform(t, p, fs=config.PRESSURE_FS) -> tuple[np.ndarray, np.ndarray]`
  - `pressure.oscillometric(t_u, p_u) -> dict` с ключами `t`, `pressure`, `envelope`, `beat_times`, `beat_amps`, `map_t`, `map_mmhg`, `deflation_rate`
  - `pressure.pressure_at(t_u, p_u, t) -> float` — давление в мм рт. ст. в момент `t`

- [ ] **Step 1: Написать падающие тесты**

```python
# tests/test_pressure.py
import numpy as np, pytest
from src import pressure, config

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
    p[: int(0.4 * 100)] += 40.0            # резкий сброс клапана
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
```

- [ ] **Step 2: Убедиться, что тесты падают**

Run: `python -m pytest tests/test_pressure.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.pressure'`

- [ ] **Step 3: Реализовать `src/pressure.py`**

```python
# src/pressure.py
"""Канал давления манжеты: чтение и осциллометрическая огибающая.

Две вещи, установленные на реальных данных и потому зашитые в код:
1. Первые PRESSURE_SKIP_S секунд отбрасываются. Сброс клапана после накачки
   даёт ступень, на которой полосовой фильтр звенит; без этого максимум
   огибающей всегда ложно попадает на первую секунду.
2. Огибающая строится по амплитуде каждой отдельной пульсации, а не скользящим
   окном фиксированной ширины — окно смазывает колокол и смещает максимум.
"""
from pathlib import Path
import numpy as np
from scipy import signal
from . import config

def read_pressure(path):
    d = np.loadtxt(Path(path), delimiter=",", ndmin=2)
    if d.shape[1] < 2:
        raise ValueError(f"ожидалось две колонки: {path}")
    return d[:, 0].astype(np.float64), d[:, 1].astype(np.float64)

def resample_uniform(t, p, fs=config.PRESSURE_FS):
    tu = np.arange(t[0], t[-1], 1.0 / fs)
    return tu, np.interp(tu, t, p)

def oscillometric(t_u, p_u, fs=config.PRESSURE_FS):
    skip = int(config.PRESSURE_SKIP_S * fs)
    sos = signal.butter(3, list(config.PRESSURE_BAND), btype="band", fs=fs, output="sos")
    osc = signal.sosfiltfilt(sos, p_u)
    osc[:skip] = 0.0

    dist = int(config.IBI_MIN_S * fs)
    peaks, _ = signal.find_peaks(osc, distance=dist)
    troughs, _ = signal.find_peaks(-osc, distance=dist)
    peaks = peaks[peaks >= skip]
    troughs = troughs[troughs >= skip]

    beat_t, beat_a = [], []
    for pk in peaks:
        before = troughs[troughs < pk]
        after = troughs[troughs > pk]
        if len(before) == 0 or len(after) == 0:
            continue
        amp = osc[pk] - 0.5 * (osc[before[-1]] + osc[after[0]])
        if amp > 0:
            beat_t.append(t_u[pk])
            beat_a.append(amp)
    beat_t = np.asarray(beat_t)
    beat_a = np.asarray(beat_a)

    env = np.zeros_like(t_u)
    if len(beat_t) >= 3:
        env = np.interp(t_u, beat_t, beat_a, left=0.0, right=0.0)
        w = int(2.0 * fs) | 1
        if len(env) > w:
            env = signal.savgol_filter(env, w, 2)
        env = np.clip(env, 0.0, None)
    env[:skip] = 0.0

    imax = int(np.argmax(env)) if env.max() > 0 else skip
    valid = slice(skip, max(skip + 2, len(p_u) - 2))
    rate = float(np.median(np.gradient(p_u, 1.0 / fs)[valid]))

    return {
        "t": t_u,
        "pressure": p_u,
        "envelope": env,
        "beat_times": beat_t,
        "beat_amps": beat_a,
        "map_t": float(t_u[imax]),
        "map_mmhg": float(p_u[imax]),
        "deflation_rate": rate,
    }

def pressure_at(t_u, p_u, t):
    return float(np.interp(t, t_u, p_u))
```

- [ ] **Step 4: Убедиться, что тесты проходят**

Run: `python -m pytest tests/test_pressure.py -v`
Expected: PASS (тест с меткой `data` пройдёт или пропустится)

- [ ] **Step 5: Коммит**

```bash
git add -A
git commit -m "feat: канал давления и осциллометрическая огибающая"
```

---

### Task 4: Манифест данных

**Files:**
- Create: `src/manifest.py`
- Test: `tests/test_manifest.py`
- Output: `data/manifest.csv`

**Interfaces:**
- Consumes: `audio.load_8k`, `pressure.read_pressure`, `config.*`
- Produces:
  - `manifest.recording_id(path) -> int` — номер из `recording (123).wav`
  - `manifest.build() -> list[dict]`
  - `manifest.write(rows, path=config.DATA_DIR/'manifest.csv') -> None`
  - `manifest.read(path=...) -> list[dict]`
  - Колонки: `id, wav_path, pressure_path, dur_s, sr_in, dup_even, dup_odd, peak, clip_frac, hf_frac, has_pressure, press_dur_s, deflation_rate, status`
  - `status` ∈ `{ok, excluded, unreadable, no_pressure, suspect}`

- [ ] **Step 1: Написать падающие тесты**

```python
# tests/test_manifest.py
import pytest
from src import manifest, config

def test_recording_id_extracts_number():
    assert manifest.recording_id("/x/recording (123).wav") == 123
    assert manifest.recording_id("recording (7).wav") == 7

def test_recording_id_rejects_unexpected_name():
    with pytest.raises(ValueError):
        manifest.recording_id("REC0.WAV")

def test_write_then_read_roundtrip(tmp_path):
    rows = [{"id": 1, "wav_path": "/a.wav", "pressure_path": "", "dur_s": 23.5,
             "sr_in": 16000, "dup_even": 1.0, "dup_odd": 0.31, "peak": 588.0,
             "clip_frac": 0.0, "hf_frac": 0.0002, "has_pressure": 0,
             "press_dur_s": "", "deflation_rate": "", "status": "no_pressure"}]
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
    assert by("unreadable") == 1                       # recording (221).wav
    assert by("excluded") == 1                         # recording (199).wav
    assert sum(1 for r in rows if r["has_pressure"]) == 784
    ok = [r for r in rows if r["status"] == "ok"]
    assert len(ok) > 750
```

- [ ] **Step 2: Убедиться, что тесты падают**

Run: `python -m pytest tests/test_manifest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.manifest'`

- [ ] **Step 3: Реализовать `src/manifest.py`**

```python
# src/manifest.py
import csv, re
from pathlib import Path
from . import audio, pressure, config

FIELDS = ["id", "wav_path", "pressure_path", "dur_s", "sr_in", "dup_even", "dup_odd",
          "peak", "clip_frac", "hf_frac", "has_pressure", "press_dur_s",
          "deflation_rate", "status"]

_NUM = re.compile(r"recording \((\d+)\)\.wav$", re.IGNORECASE)

def recording_id(path):
    m = _NUM.search(str(path))
    if not m:
        raise ValueError(f"неожиданное имя файла: {path}")
    return int(m.group(1))

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

        row.update(dur_s=round(diag["dur_s"], 3), sr_in=diag["sr_in"],
                   dup_even=round(diag["dup_even"], 4), dup_odd=round(diag["dup_odd"], 4),
                   peak=diag["peak"], clip_frac=round(diag["clip_frac"], 6),
                   hf_frac=round(diag["hf_frac"], 6))

        if rid in config.EXCLUDED:
            row["status"] = "excluded"

        pcsv = config.pressure_dir() / f"pressure {rid}.csv"
        if pcsv.exists():
            try:
                t, p = pressure.read_pressure(pcsv)
                tu, pu = pressure.resample_uniform(t, p)
                r = pressure.oscillometric(tu, pu)
                row.update(pressure_path=str(pcsv), has_pressure=1,
                           press_dur_s=round(float(t[-1]), 3),
                           deflation_rate=round(r["deflation_rate"], 3))
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

_NUMERIC = {"id": int, "sr_in": int, "has_pressure": int, "dur_s": float,
            "dup_even": float, "dup_odd": float, "peak": float, "clip_frac": float,
            "hf_frac": float, "press_dur_s": float, "deflation_rate": float}

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
    rows = build()
    write(rows)
    from collections import Counter
    print(f"записей: {len(rows)}")
    for k, v in Counter(r["status"] for r in rows).most_common():
        print(f"  {k}: {v}")
    print(f"  с давлением: {sum(r['has_pressure'] for r in rows)}")
```

- [ ] **Step 4: Убедиться, что тесты проходят**

Run: `python -m pytest tests/test_manifest.py -v`
Expected: PASS

- [ ] **Step 5: Построить манифест на реальных данных**

Run: `python -m src.manifest`
Expected: 906 записей, `unreadable: 1`, `excluded: 1`, с давлением 784.

- [ ] **Step 6: Коммит**

```bash
git add -A
git commit -m "feat: манифест данных с диагностикой качества записей"
```

---

### Task 5: Синхронизация аудио и давления — критическая точка

**Files:**
- Create: `src/sync.py`
- Test: `tests/test_sync.py`
- Output: `reports/sync.md`

Это единственная задача, результат которой может отменить план. Если синхронизация не подтвердится, канал давления перестаёт быть источником меток. **Не начинать Task 6, не показав результат человеку.**

**Interfaces:**
- Consumes: `audio.load_8k`, `audio.features`, `pressure.*`, `manifest.read`
- Produces:
  - `sync.audio_beat_times(x) -> np.ndarray` — моменты ударов в звуке, секунды
  - `sync.estimate_offset(audio_beats, press_beats, max_lag=6.0, step=0.02) -> tuple[float, float]` — `(offset_s, score)`, где score ∈ 0..1
  - `sync.estimate_all(rows) -> list[dict]` с ключами `id, offset, score`
  - `sync.report(results) -> str` — markdown

Смысл `offset`: `t_давления = t_аудио + offset`.

- [ ] **Step 1: Написать падающие тесты**

```python
# tests/test_sync.py
import numpy as np
from src import sync

def test_estimate_offset_recovers_known_shift():
    rng = np.random.default_rng(0)
    beats = np.cumsum(rng.uniform(0.75, 0.95, 30))
    shift = 1.4
    off, score = sync.estimate_offset(beats, beats + shift)
    assert abs(off - shift) < 0.1
    assert score > 0.7

def test_estimate_offset_survives_missing_beats():
    rng = np.random.default_rng(1)
    beats = np.cumsum(rng.uniform(0.75, 0.95, 40))
    partial = np.delete(beats, rng.choice(40, 12, replace=False)) + 0.8
    off, score = sync.estimate_offset(beats, partial)
    assert abs(off - 0.8) < 0.15
    assert score > 0.4

def test_estimate_offset_reports_low_score_on_unrelated_sequences():
    rng = np.random.default_rng(2)
    a = np.cumsum(rng.uniform(0.7, 1.0, 30))
    b = np.cumsum(rng.uniform(0.4, 0.5, 30))
    off, score = sync.estimate_offset(a, b)
    assert score < 0.5
```

- [ ] **Step 2: Убедиться, что тесты падают**

Run: `python -m pytest tests/test_sync.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.sync'`

- [ ] **Step 3: Реализовать `src/sync.py`**

Сопоставляются моменты ударов, а не амплитуды: кросс-корреляция огибающих даёт корреляцию 0.31–0.53 и разброс ±2.3 с — это проверено и признано недостаточным.

```python
# src/sync.py
import numpy as np
from scipy import signal
from . import audio, pressure, config, manifest

def audio_beat_times(x, sr=config.SR):
    env = audio.rms_envelope(audio.bandpass(x, config.BAND_LO, config.BAND_HI, sr), sr)
    if len(env) < 10:
        return np.zeros(0)
    dist = int(config.IBI_MIN_S / config.HOP_S)
    prom = float(np.percentile(env, 75) - np.percentile(env, 25))
    peaks, _ = signal.find_peaks(env, distance=dist, prominence=max(prom * 0.5, 1e-9))
    return peaks * config.HOP_S

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
    pb = np.sort(np.asarray(press_beats))
    ab = np.sort(np.asarray(audio_beats))
    for lag in lags:
        shifted = ab + lag
        idx = np.searchsorted(pb, shifted)
        idx = np.clip(idx, 1, len(pb) - 1)
        left = np.abs(pb[idx - 1] - shifted)
        right = np.abs(pb[idx] - shifted)
        matched = int(np.sum(np.minimum(left, right) <= tol))
        score = matched / denom
        if score > best_score:
            best_score, best_lag = score, float(lag)
    return best_lag, best_score

def estimate_all(rows=None):
    rows = rows or [r for r in manifest.read() if r["status"] in ("ok", "suspect") and r["has_pressure"]]
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

ACCEPT_MIN_SHARE = 0.70
ACCEPT_MAX_SPREAD = 0.30

def report(results):
    good = [r for r in results if r["score"] >= 0.5 and np.isfinite(r["offset"])]
    offs = np.array([r["offset"] for r in good]) if good else np.zeros(0)
    lines = ["# Синхронизация аудио и давления", ""]
    lines.append(f"- файлов проверено: {len(results)}")
    lines.append(f"- с уверенным совпадением (score >= 0.5): {len(good)} "
                 f"({100*len(good)/max(len(results),1):.1f}%)")
    verdict = "НЕ ПОДТВЕРЖДЕНА"
    if len(offs):
        med = float(np.median(offs))
        within = float(np.mean(np.abs(offs - med) <= ACCEPT_MAX_SPREAD))
        lines += [f"- медианный сдвиг: {med:+.3f} с",
                  f"- в пределах ±{ACCEPT_MAX_SPREAD} с от медианы: {100*within:.1f}%",
                  f"- разброс: p10={np.percentile(offs,10):+.2f} p90={np.percentile(offs,90):+.2f}"]
        share = len(good) / max(len(results), 1)
        if share >= ACCEPT_MIN_SHARE and within >= ACCEPT_MIN_SHARE:
            verdict = f"ПОДТВЕРЖДЕНА, сдвиг {med:+.3f} с"
        elif within >= ACCEPT_MIN_SHARE:
            verdict = (f"ЧАСТИЧНО: сдвиг устойчив ({med:+.3f} с), но уверенно совпало "
                       f"лишь {100*share:.0f}% файлов")
    lines += ["", f"## Вердикт: {verdict}", "",
              "Критерий приёмки: не менее 70% файлов дают уверенное совпадение "
              "и не менее 70% оценок лежат в пределах ±0.30 с от медианы."]
    return "\n".join(lines)

if __name__ == "__main__":
    res = estimate_all()
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    text = report(res)
    (config.REPORT_DIR / "sync.md").write_text(text)
    import csv
    with (config.DATA_DIR / "sync.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "offset", "score", "n_audio", "n_press"])
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in w.fieldnames} for r in res])
    print(text)
```

- [ ] **Step 4: Убедиться, что тесты проходят**

Run: `python -m pytest tests/test_sync.py -v`
Expected: PASS, 3 теста

- [ ] **Step 5: Запустить на всех парах и прочитать вердикт**

Run: `python -m src.sync`
Expected: отчёт в `reports/sync.md` с одним из трёх вердиктов.

- [ ] **Step 6: ОСТАНОВКА — показать вердикт человеку**

Не переходить к Task 6 самостоятельно. Возможные исходы:

| Вердикт | Что делать |
|---|---|
| ПОДТВЕРЖДЕНА | Продолжать. Значение сдвига внести в `config.SYNC_OFFSET_S`. |
| ЧАСТИЧНО | Продолжать, но автометки строить только для файлов со `score >= 0.5`, остальные — в ручную очередь. |
| НЕ ПОДТВЕРЖДЕНА | План Б: Task 6 сокращается до меток из звука, объём ручной проверки растёт до ~400 файлов. |

- [ ] **Step 7: Коммит**

```bash
git add -A
git commit -m "feat: оценка сдвига между осями аудио и давления"
```

---

### Task 6: Автоматическая разметка

**Files:**
- Create: `src/autolabel.py`
- Test: `tests/test_autolabel.py`
- Output: `data/labels_auto.csv`

**Interfaces:**
- Consumes: `audio.*`, `pressure.*`, `sync.*`, `manifest.read`, `config.*`
- Produces:
  - `autolabel.interval_from_pressure(osc, rise=0.5, fall=0.7) -> tuple[float, float] | None` — в осях давления
  - `autolabel.interval_from_audio(x) -> tuple[float, float] | None` — в осях аудио
  - `autolabel.combine(p_iv, a_iv, offset) -> dict` с ключами `start, end, disagreement, priority, source`
  - `autolabel.build(rows, offsets) -> list[dict]`
  - Колонки CSV: `id, start, end, p_start, p_end, a_start, a_end, disagreement, priority, source`

`priority` — число: чем больше, тем раньше файл покажут человеку. Растёт с расхождением источников и с отсутствием одного из них.

- [ ] **Step 1: Написать падающие тесты**

```python
# tests/test_autolabel.py
import numpy as np
from src import autolabel, pressure, config

def _osc_with_bell(peak_t=12.0, width=3.5, dur=25.0):
    fs = config.PRESSURE_FS
    t = np.arange(0, dur, 1 / fs)
    env = np.exp(-0.5 * ((t - peak_t) / width) ** 2)
    env[: int(config.PRESSURE_SKIP_S * fs)] = 0.0
    return {"t": t, "pressure": 160 - 4.8 * t, "envelope": env,
            "beat_times": np.arange(2, dur, 0.85), "beat_amps": np.ones(1),
            "map_t": peak_t, "map_mmhg": 160 - 4.8 * peak_t, "deflation_rate": -4.8}

def test_interval_from_pressure_brackets_the_bell():
    iv = autolabel.interval_from_pressure(_osc_with_bell(peak_t=12.0, width=3.5))
    assert iv is not None
    s, e = iv
    assert s < 12.0 < e
    assert 2.0 < e - s < 18.0

def test_interval_from_pressure_returns_none_on_flat_envelope():
    osc = _osc_with_bell()
    osc["envelope"] = np.zeros_like(osc["envelope"])
    assert autolabel.interval_from_pressure(osc) is None

def test_combine_flags_disagreement():
    r = autolabel.combine((5.0, 15.0), (5.2, 15.1), offset=0.0)
    assert r["disagreement"] < 0.5
    assert r["priority"] < 1.0

    r2 = autolabel.combine((5.0, 15.0), (9.0, 20.0), offset=0.0)
    assert r2["disagreement"] > 3.0
    assert r2["priority"] > r["priority"]

def test_combine_applies_offset():
    """t_давления = t_аудио + offset, значит метки давления сдвигаются назад."""
    r = autolabel.combine((6.0, 16.0), None, offset=2.0)
    assert abs(r["start"] - 4.0) < 1e-6
    assert abs(r["end"] - 14.0) < 1e-6
    assert r["source"] == "pressure"

def test_combine_uses_audio_when_pressure_missing():
    r = autolabel.combine(None, (3.0, 11.0), offset=1.0)
    assert (r["start"], r["end"]) == (3.0, 11.0)
    assert r["source"] == "audio"
    assert r["priority"] > 1.0

def test_combine_returns_none_when_both_missing():
    assert autolabel.combine(None, None, offset=0.0) is None
```

- [ ] **Step 2: Убедиться, что тесты падают**

Run: `python -m pytest tests/test_autolabel.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.autolabel'`

- [ ] **Step 3: Реализовать `src/autolabel.py`**

```python
# src/autolabel.py
"""Метки из двух независимых источников с последующим согласованием.

Коэффициенты RISE/FALL — начальные значения, а не окончательные. Их надо
перекалибровать на калибровочном подмножестве проверенных файлов (Task 8),
и только по нему: тестовые файлы для этого использовать нельзя.
"""
import csv
import numpy as np
from scipy import signal
from . import audio, pressure, config, manifest

RISE = 0.50   # доля максимума огибающей на подъёме -> начало
FALL = 0.70   # доля максимума на спаде -> конец

def interval_from_pressure(osc, rise=RISE, fall=FALL):
    env, t = osc["envelope"], osc["t"]
    if env.max() <= 0:
        return None
    a = env / env.max()
    imax = int(np.argmax(a))
    up = np.where(a[:imax] >= rise)[0]
    dn = np.where(a[imax:] <= fall)[0]
    if len(up) == 0 or len(dn) == 0:
        return None
    return float(t[up[0]]), float(t[imax + dn[0]])

def interval_from_audio(x, sr=config.SR):
    env = audio.rms_envelope(audio.bandpass(x, config.BAND_LO, config.BAND_HI, sr), sr)
    per = audio.periodicity(env)
    if len(env) < 20:
        return None
    w = int(2.0 / config.HOP_S)
    pad = np.pad(env.astype(np.float64), (w // 2, w // 2), mode="edge")
    loud = np.array([np.percentile(pad[i:i + w], 90) for i in range(len(env))])
    loud = loud / (np.percentile(loud, 97) + 1e-12)
    score = per * np.clip(loud, 0, 1.2)
    score = signal.medfilt(score, kernel_size=int(0.5 / config.HOP_S) | 1)

    thr = max(0.15, 0.4 * float(np.percentile(score, 95)))
    mask = score > thr
    if not mask.any():
        return None
    idx = np.where(mask)[0]
    return float(idx[0] * config.HOP_S), float(idx[-1] * config.HOP_S)

def combine(p_iv, a_iv, offset):
    """Сводит два источника. p_iv задан в осях давления, a_iv — в осях аудио."""
    p_audio = None if p_iv is None else (p_iv[0] - offset, p_iv[1] - offset)

    if p_audio is None and a_iv is None:
        return None
    if p_audio is None:
        return {"start": a_iv[0], "end": a_iv[1], "disagreement": float("nan"),
                "priority": 2.0, "source": "audio"}
    if a_iv is None:
        return {"start": p_audio[0], "end": p_audio[1], "disagreement": float("nan"),
                "priority": 2.0, "source": "pressure"}

    dis = max(abs(p_audio[0] - a_iv[0]), abs(p_audio[1] - a_iv[1]))
    start = 0.5 * (p_audio[0] + a_iv[0])
    end = 0.5 * (p_audio[1] + a_iv[1])
    return {"start": float(start), "end": float(end), "disagreement": float(dis),
            "priority": float(dis / 1.5), "source": "both"}

FIELDS = ["id", "start", "end", "p_start", "p_end", "a_start", "a_end",
          "disagreement", "priority", "source"]

def build(rows=None, offsets=None):
    rows = rows or [r for r in manifest.read() if r["status"] in ("ok", "suspect")]
    offsets = offsets or {}
    out = []
    for r in rows:
        rid = r["id"]
        try:
            x, _ = audio.load_8k(r["wav_path"])
        except Exception:
            continue
        a_iv = interval_from_audio(x)
        p_iv = None
        if r["has_pressure"]:
            try:
                t, p = pressure.read_pressure(r["pressure_path"])
                tu, pu = pressure.resample_uniform(t, p)
                p_iv = interval_from_pressure(pressure.oscillometric(tu, pu))
            except Exception:
                p_iv = None
        c = combine(p_iv, a_iv, offsets.get(rid, config.SYNC_OFFSET_S))
        if c is None:
            continue
        dur = len(x) / config.SR
        c["start"] = float(np.clip(c["start"], 0.0, dur))
        c["end"] = float(np.clip(c["end"], 0.0, dur))
        if c["end"] - c["start"] < 1.0:
            c["priority"] += 3.0            # подозрительно короткий интервал
        out.append({"id": rid, **c,
                    "p_start": "" if p_iv is None else round(p_iv[0], 3),
                    "p_end": "" if p_iv is None else round(p_iv[1], 3),
                    "a_start": "" if a_iv is None else round(a_iv[0], 3),
                    "a_end": "" if a_iv is None else round(a_iv[1], 3)})
    return out

def write(rows, path=None):
    path = path or config.DATA_DIR / "labels_auto.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})

if __name__ == "__main__":
    import csv as _csv
    offsets = {}
    sp = config.DATA_DIR / "sync.csv"
    if sp.exists():
        with sp.open() as fh:
            for row in _csv.DictReader(fh):
                if row["score"] and float(row["score"]) >= 0.5:
                    offsets[int(row["id"])] = float(row["offset"])
    rows = build(offsets=offsets)
    write(rows)
    d = np.array([r["disagreement"] for r in rows if np.isfinite(r["disagreement"])])
    print(f"размечено: {len(rows)}")
    print(f"оба источника: {sum(1 for r in rows if r['source']=='both')}")
    if len(d):
        print(f"расхождение: медиана {np.median(d):.2f} с, p90 {np.percentile(d,90):.2f} с")
        print(f"расхождение > 1.5 с: {int(np.sum(d>1.5))} файлов -> в начало очереди проверки")
```

- [ ] **Step 4: Убедиться, что тесты проходят**

Run: `python -m pytest tests/test_autolabel.py -v`
Expected: PASS, 6 тестов

- [ ] **Step 5: Внести сдвиг из Task 5**

Записать значение из `reports/sync.md` в `config.SYNC_OFFSET_S`.

- [ ] **Step 6: Построить автометки**

Run: `python -m src.autolabel`
Expected: `data/labels_auto.csv`, статистика расхождений в консоли.

- [ ] **Step 7: Коммит**

```bash
git add -A
git commit -m "feat: автоматическая разметка из давления и звука с согласованием"
```

---

### Task 7: HTML-разметчик

**Files:**
- Create: `tools/make_annotator.py`
- Create: `tools/annotator_template.html`
- Test: `tests/test_annotator.py`
- Output: `data/annotator.html`

**Interfaces:**
- Consumes: `audio.*`, `pressure.*`, `manifest.read`, `autolabel` CSV
- Produces:
  - `make_annotator.build_payload(rows, labels, limit=250) -> dict`
  - `make_annotator.render(payload, template_path) -> str`
  - Готовый самодостаточный HTML в `data/annotator.html`

Аудио кладётся в страницу как WAV в base64, приведённый к 8 кГц и усиленный — записи тихие. 250 файлов по ~28 с на 8 кГц моно 16 бит дают около 110 МБ, поэтому в страницу пишется не сырой звук, а децимированная огибающая для рисования плюс отдельные короткие WAV по требованию.

Решение: страница рисует огибающую и кривую давления из JSON, а звук подгружает по клику через `fetch` на `file://` — это не работает из-за политики браузера. Поэтому звук встраивается сразу, но **в виде огибающей для глаз и WAV на 2000 Гц для ушей**: понижение до 2000 Гц сохраняет всю полосу тонов Короткова (20–200 Гц) и уменьшает объём в 4 раза, до ~28 МБ на 250 файлов.

- [ ] **Step 1: Написать падающие тесты**

```python
# tests/test_annotator.py
import base64, json, numpy as np
from tools import make_annotator as mk

def test_wav_bytes_roundtrip():
    x = (0.5 * np.sin(2 * np.pi * 50 * np.arange(4000) / 2000)).astype(np.float32)
    b = mk.wav_bytes(x, 2000)
    assert b[:4] == b"RIFF" and b[8:12] == b"WAVE"
    assert len(b) == 44 + len(x) * 2

def test_build_payload_sorts_by_priority_desc():
    rows = [{"id": 1, "wav_path": "a", "pressure_path": "", "has_pressure": 0, "dur_s": 10.0},
            {"id": 2, "wav_path": "b", "pressure_path": "", "has_pressure": 0, "dur_s": 10.0}]
    labels = {1: {"start": 1.0, "end": 5.0, "priority": 0.2},
              2: {"start": 2.0, "end": 6.0, "priority": 9.9}}
    p = mk.build_payload(rows, labels, limit=2, load_audio=False)
    assert [item["id"] for item in p["items"]] == [2, 1]

def test_render_produces_standalone_html(tmp_path):
    tpl = tmp_path / "t.html"
    tpl.write_text("<html><body><script>const DATA=__PAYLOAD__;</script></body></html>")
    html = mk.render({"items": [{"id": 7}]}, tpl)
    assert "__PAYLOAD__" not in html
    assert '"id": 7' in html or '"id":7' in html
    assert json.loads(html.split("const DATA=")[1].split(";</script>")[0])["items"][0]["id"] == 7
```

- [ ] **Step 2: Убедиться, что тесты падают**

Run: `python -m pytest tests/test_annotator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.make_annotator'`

- [ ] **Step 3: Реализовать генератор**

```python
# tools/__init__.py — пустой файл
```

```python
# tools/make_annotator.py
import base64, json, struct, sys
from pathlib import Path
import numpy as np
from scipy import signal
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import audio, pressure, config, manifest

PLAY_SR = 2000          # полосы 20-200 Гц хватает с запасом, объём меньше вчетверо
DRAW_HZ = 50            # точек огибающей в секунду для рисования

def wav_bytes(x, sr):
    """PCM16 WAV в память. x — float в диапазоне примерно -1..1."""
    pcm = np.clip(x * 32767.0, -32768, 32767).astype("<i2").tobytes()
    hdr = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE"
    hdr += b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
    hdr += b"data" + struct.pack("<I", len(pcm))
    return hdr + pcm

def _playable(x):
    y = audio.bandpass(x, config.BAND_LO, config.BAND_HI)
    y = signal.resample_poly(y, PLAY_SR, config.SR)
    peak = float(np.max(np.abs(y))) or 1.0
    return (y / peak * 0.9).astype(np.float32)

def _draw_envelope(x):
    env = audio.rms_envelope(audio.bandpass(x, config.BAND_LO, config.BAND_HI))
    step = max(1, int(round((1 / config.HOP_S) / DRAW_HZ)))
    e = env[: len(env) // step * step].reshape(-1, step).max(axis=1)
    return (e / (e.max() + 1e-12)).round(4).tolist()

def build_payload(rows, labels, limit=250, load_audio=True):
    by_priority = sorted(rows, key=lambda r: -labels.get(r["id"], {}).get("priority", 0.0))
    items = []
    for r in by_priority[:limit]:
        lab = labels.get(r["id"], {})
        item = {"id": r["id"], "dur": float(r["dur_s"]),
                "start": float(lab.get("start", 0.0)), "end": float(lab.get("end", 0.0)),
                "priority": float(lab.get("priority", 0.0)),
                "source": lab.get("source", ""), "env": [], "press": [], "audio": ""}
        if load_audio:
            x, _ = audio.load_8k(r["wav_path"])
            item["dur"] = len(x) / config.SR
            item["env"] = _draw_envelope(x)
            item["audio"] = base64.b64encode(wav_bytes(_playable(x), PLAY_SR)).decode()
            if r["has_pressure"] and r["pressure_path"]:
                t, p = pressure.read_pressure(r["pressure_path"])
                tu, pu = pressure.resample_uniform(t, p)
                k = max(1, int(len(tu) / (item["dur"] * DRAW_HZ)))
                item["press"] = [[round(float(a), 2), round(float(b), 1)]
                                 for a, b in zip(tu[::k], pu[::k])]
        items.append(item)
    return {"items": items, "play_sr": PLAY_SR, "draw_hz": DRAW_HZ}

def render(payload, template_path):
    tpl = Path(template_path).read_text()
    return tpl.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False))

if __name__ == "__main__":
    import csv
    rows = [r for r in manifest.read() if r["status"] in ("ok", "suspect")]
    labels = {}
    with (config.DATA_DIR / "labels_auto.csv").open() as fh:
        for row in csv.DictReader(fh):
            labels[int(row["id"])] = {
                "start": float(row["start"]), "end": float(row["end"]),
                "priority": float(row["priority"]), "source": row["source"]}
    payload = build_payload(rows, labels, limit=250)
    out = config.DATA_DIR / "annotator.html"
    out.write_text(render(payload, Path(__file__).parent / "annotator_template.html"))
    print(f"готово: {out}  ({out.stat().st_size/1e6:.1f} МБ, {len(payload['items'])} записей)")
    print("Открыть в браузере, выверить метки, нажать «Сохранить CSV».")
```

- [ ] **Step 4: Написать шаблон `tools/annotator_template.html`**

```html
<!doctype html>
<meta charset="utf-8">
<title>Разметка интервала тонов Короткова</title>
<style>
  body { font: 14px/1.5 system-ui, sans-serif; margin: 0; padding: 16px; background:#111; color:#eee; }
  #bar { display:flex; gap:16px; align-items:center; margin-bottom:12px; flex-wrap:wrap; }
  button { background:#2a2a2a; color:#eee; border:1px solid #444; padding:6px 12px; border-radius:6px; cursor:pointer; }
  button:hover { background:#3a3a3a; }
  #cv { width:100%; height:320px; display:block; background:#181818; border:1px solid #333; border-radius:6px; cursor:ew-resize; }
  .num { font-variant-numeric: tabular-nums; }
  #done { color:#7c7; }
  kbd { background:#333; border-radius:3px; padding:1px 5px; font-size:12px; }
</style>

<div id="bar">
  <button id="prev">← Назад</button>
  <button id="next">Вперёд →</button>
  <button id="play">▶ Пробел</button>
  <button id="accept">✓ Принять (Enter)</button>
  <button id="reset">↺ Вернуть авто (r)</button>
  <button id="save">Сохранить CSV</button>
  <span id="pos" class="num"></span>
  <span id="done" class="num"></span>
</div>
<canvas id="cv"></canvas>
<p><span id="meta" class="num"></span> &nbsp; <kbd>←</kbd><kbd>→</kbd> файлы,
   <kbd>пробел</kbd> звук, <kbd>Enter</kbd> принять, <kbd>r</kbd> сброс.
   Метки тянутся мышью.</p>

<script>
const DATA = __PAYLOAD__;
const items = DATA.items;
const KEY = "korotkoff-annotations-v1";
let state = JSON.parse(localStorage.getItem(KEY) || "{}");
let idx = 0, drag = null, audio = null;

const cv = document.getElementById("cv"), ctx = cv.getContext("2d");

function cur() { return items[idx]; }
function marks() {
  const it = cur(), st = state[it.id];
  return st ? {start: st.start, end: st.end, verified: st.verified}
            : {start: it.start, end: it.end, verified: false};
}
function setMarks(m) {
  state[cur().id] = {start: m.start, end: m.end, verified: m.verified};
  localStorage.setItem(KEY, JSON.stringify(state));
}

function resize() {
  const r = cv.getBoundingClientRect();
  cv.width = r.width * devicePixelRatio;
  cv.height = r.height * devicePixelRatio;
  ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  draw();
}
addEventListener("resize", resize);

const t2x = t => (t / cur().dur) * cv.width / devicePixelRatio;
const x2t = x => (x / (cv.width / devicePixelRatio)) * cur().dur;

function draw() {
  const it = cur(), m = marks();
  const W = cv.width / devicePixelRatio, H = cv.height / devicePixelRatio;
  ctx.clearRect(0, 0, W, H);

  // выделение интервала
  ctx.fillStyle = "rgba(90,170,255,0.13)";
  ctx.fillRect(t2x(m.start), 0, t2x(m.end) - t2x(m.start), H);

  // огибающая звука, верхние две трети
  const envH = H * 0.62;
  ctx.strokeStyle = "#7fd67f"; ctx.lineWidth = 1; ctx.beginPath();
  it.env.forEach((v, i) => {
    const x = (i / (it.env.length - 1)) * W, y = envH - v * (envH - 10);
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.stroke();

  // кривая давления, нижняя треть
  if (it.press && it.press.length) {
    const ps = it.press.map(p => p[1]);
    const lo = Math.min(...ps), hi = Math.max(...ps);
    ctx.strokeStyle = "#e0a060"; ctx.beginPath();
    it.press.forEach((p, i) => {
      const x = t2x(p[0]), y = H - 8 - ((p[1] - lo) / (hi - lo + 1e-9)) * (H * 0.32);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
    ctx.fillStyle = "#e0a060"; ctx.font = "11px sans-serif";
    ctx.fillText(hi.toFixed(0) + " мм рт. ст.", 4, H * 0.70);
    ctx.fillText(lo.toFixed(0), 4, H - 10);
  }

  // метки
  [["start", "#4af"], ["end", "#f66"]].forEach(([k, col]) => {
    const x = t2x(m[k]);
    ctx.strokeStyle = col; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    ctx.fillStyle = col; ctx.font = "12px sans-serif";
    ctx.fillText(m[k].toFixed(2) + " с", x + 4, 14);
  });

  document.getElementById("pos").textContent = `${idx + 1} / ${items.length}`;
  document.getElementById("done").textContent =
    `проверено ${Object.values(state).filter(v => v.verified).length}`;
  document.getElementById("meta").textContent =
    `id ${it.id} · ${it.dur.toFixed(1)} с · источник ${it.source} · ` +
    `приоритет ${it.priority.toFixed(2)} · длина ${(m.end - m.start).toFixed(2)} с` +
    (m.verified ? " · ПРОВЕРЕНО" : "");
}

cv.addEventListener("mousedown", e => {
  const x = e.offsetX, m = marks();
  drag = Math.abs(x - t2x(m.start)) < Math.abs(x - t2x(m.end)) ? "start" : "end";
});
addEventListener("mouseup", () => { drag = null; });
cv.addEventListener("mousemove", e => {
  if (!drag) return;
  const m = marks();
  m[drag] = Math.max(0, Math.min(cur().dur, x2t(e.offsetX)));
  if (m.start > m.end) { const t = m.start; m.start = m.end; m.end = t; }
  setMarks(m); draw();
});

function load(i) {
  idx = (i + items.length) % items.length;
  if (audio) { audio.pause(); audio = null; }
  draw();
}
function play() {
  if (audio) { audio.pause(); audio = null; return; }
  audio = new Audio("data:audio/wav;base64," + cur().audio);
  audio.currentTime = Math.max(0, marks().start - 0.5);
  audio.play();
}

document.getElementById("prev").onclick = () => load(idx - 1);
document.getElementById("next").onclick = () => load(idx + 1);
document.getElementById("play").onclick = play;
document.getElementById("accept").onclick = () => {
  const m = marks(); m.verified = true; setMarks(m); load(idx + 1);
};
document.getElementById("reset").onclick = () => {
  delete state[cur().id];
  localStorage.setItem(KEY, JSON.stringify(state)); draw();
};
document.getElementById("save").onclick = () => {
  const lines = ["id,start,end,verified"];
  items.forEach(it => {
    const st = state[it.id];
    if (st && st.verified)
      lines.push(`${it.id},${st.start.toFixed(3)},${st.end.toFixed(3)},1`);
  });
  const url = URL.createObjectURL(new Blob([lines.join("\n") + "\n"], {type: "text/csv"}));
  const a = document.createElement("a");
  a.href = url; a.download = "labels_verified.csv"; a.click();
  URL.revokeObjectURL(url);
};

addEventListener("keydown", e => {
  if (e.key === "ArrowLeft") load(idx - 1);
  else if (e.key === "ArrowRight") load(idx + 1);
  else if (e.key === " ") { e.preventDefault(); play(); }
  else if (e.key === "Enter") document.getElementById("accept").click();
  else if (e.key === "r") document.getElementById("reset").click();
});

resize();
</script>
```

Прогресс держится в `localStorage`, поэтому случайное закрытие вкладки не теряет работу.
Сохранённый CSV содержит только записи, помеченные <kbd>Enter</kbd>.

- [ ] **Step 5: Убедиться, что тесты проходят**

Run: `python -m pytest tests/test_annotator.py -v`
Expected: PASS, 3 теста

- [ ] **Step 6: Сгенерировать и проверить размер**

Run: `python -m tools.make_annotator`
Expected: `data/annotator.html` размером примерно 25–35 МБ. Если больше 60 МБ — уменьшить `limit` или `PLAY_SR`.

- [ ] **Step 7: Коммит**

```bash
git add -A
git commit -m "feat: автономный HTML-разметчик с приоритетной очередью"
```

- [ ] **Step 8: ОСТАНОВКА — человек размечает**

Открыть `data/annotator.html`, выверить примерно 200 записей, сохранить `data/labels_verified.csv`. Дальнейшие задачи опираются на этот файл.

---

### Task 8: Разбиение выборки и калибровка коэффициентов

**Files:**
- Create: `src/splits.py`
- Test: `tests/test_splits.py`
- Output: `data/splits.json`, обновлённые `RISE`/`FALL` в `src/autolabel.py`

**Interfaces:**
- Consumes: `manifest.read`, `data/labels_verified.csv`
- Produces:
  - `splits.block_split(ids, fractions=(0.7,0.15,0.15)) -> dict[str, list[int]]`
  - `splits.split_verified(verified_ids, calib_frac=0.3) -> tuple[list[int], list[int]]`
  - `splits.load() -> dict` / `splits.save(d) -> None`
  - `splits.calibrate_ratios(calib_ids) -> tuple[float, float]`

- [ ] **Step 1: Написать падающие тесты**

```python
# tests/test_splits.py
import pytest
from src import splits

def test_block_split_keeps_ids_contiguous():
    ids = list(range(1, 101))
    d = splits.block_split(ids, (0.7, 0.15, 0.15))
    assert d["train"] == list(range(1, 71))
    assert d["val"] == list(range(71, 86))
    assert d["test"] == list(range(86, 101))

def test_block_split_never_interleaves():
    """Случайное разбиение дало бы утечку: соседние номера — один испытуемый."""
    ids = list(range(1, 51))
    d = splits.block_split(ids)
    assert max(d["train"]) < min(d["val"])
    assert max(d["val"]) < min(d["test"])

def test_block_split_covers_every_id_exactly_once():
    ids = [3, 9, 14, 27, 31, 55, 78, 90]
    d = splits.block_split(ids)
    assert sorted(d["train"] + d["val"] + d["test"]) == sorted(ids)

def test_split_verified_is_disjoint():
    ids = list(range(1, 201))
    calib, test = splits.split_verified(ids, calib_frac=0.3)
    assert set(calib).isdisjoint(test)
    assert len(calib) + len(test) == 200
    assert 55 <= len(calib) <= 65
    assert max(calib) < min(test)
```

- [ ] **Step 2: Убедиться, что тесты падают**

Run: `python -m pytest tests/test_splits.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.splits'`

- [ ] **Step 3: Реализовать `src/splits.py`**

```python
# src/splits.py
"""Разбиение только непрерывными блоками номеров.

Случайное разбиение запрещено: соседние номера файлов почти наверняка
принадлежат одному испытуемому, и случайное деление даёт утечку с
завышенной оценкой качества.
"""
import csv, json
import numpy as np
from . import config

def block_split(ids, fractions=(0.7, 0.15, 0.15)):
    ids = sorted(ids)
    n = len(ids)
    n_tr = int(round(n * fractions[0]))
    n_va = int(round(n * fractions[1]))
    return {"train": ids[:n_tr], "val": ids[n_tr:n_tr + n_va], "test": ids[n_tr + n_va:]}

def split_verified(verified_ids, calib_frac=0.3):
    """Калибровочные файлы настраивают коэффициенты и порог; тестовые не трогает ничто."""
    ids = sorted(verified_ids)
    k = int(round(len(ids) * calib_frac))
    return ids[:k], ids[k:]

def read_verified(path=None):
    path = path or config.DATA_DIR / "labels_verified.csv"
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("verified", "1") in ("1", "true", "True"):
                out[int(row["id"])] = (float(row["start"]), float(row["end"]))
    return out

def calibrate_ratios(calib_ids, verified=None):
    """Подбирает RISE/FALL так, чтобы метки из давления легли на проверенные вручную."""
    from . import autolabel, pressure, manifest
    verified = verified or read_verified()
    rows = {r["id"]: r for r in manifest.read()}
    cache = {}
    for rid in calib_ids:
        r = rows.get(rid)
        if not r or not r["has_pressure"]:
            continue
        t, p = pressure.read_pressure(r["pressure_path"])
        tu, pu = pressure.resample_uniform(t, p)
        cache[rid] = pressure.oscillometric(tu, pu)

    best, best_err = (autolabel.RISE, autolabel.FALL), float("inf")
    for rise in np.arange(0.30, 0.75, 0.05):
        for fall in np.arange(0.40, 0.95, 0.05):
            errs = []
            for rid, osc in cache.items():
                iv = autolabel.interval_from_pressure(osc, rise, fall)
                if iv is None:
                    continue
                gs, ge = verified[rid]
                errs.append(abs(iv[0] - config.SYNC_OFFSET_S - gs))
                errs.append(abs(iv[1] - config.SYNC_OFFSET_S - ge))
            if errs and np.median(errs) < best_err:
                best_err, best = float(np.median(errs)), (float(rise), float(fall))
    return best, best_err

def save(d, path=None):
    path = path or config.DATA_DIR / "splits.json"
    with open(path, "w") as fh:
        json.dump(d, fh, indent=2)

def load(path=None):
    path = path or config.DATA_DIR / "splits.json"
    with open(path) as fh:
        return json.load(fh)

if __name__ == "__main__":
    from . import manifest
    verified = read_verified()
    calib, test_v = split_verified(list(verified))
    usable = [r["id"] for r in manifest.read()
              if r["status"] in ("ok", "suspect") and r["id"] not in test_v]
    d = block_split(usable)
    d["verified_calib"] = calib
    d["verified_test"] = test_v
    save(d)
    (rise, fall), err = calibrate_ratios(calib, verified)
    print(f"train/val/test: {len(d['train'])}/{len(d['val'])}/{len(d['test'])}")
    print(f"проверено вручную: калибровка {len(calib)}, тест {len(test_v)}")
    print(f"калиброванные коэффициенты: RISE={rise:.2f} FALL={fall:.2f}, "
          f"медианная ошибка границы {err:.2f} с")
    print("Внести значения в src/autolabel.py и перегенерировать labels_auto.csv.")
```

- [ ] **Step 4: Убедиться, что тесты проходят**

Run: `python -m pytest tests/test_splits.py -v`
Expected: PASS, 4 теста

- [ ] **Step 5: Построить разбиение и откалибровать**

Run: `python -m src.splits`
Затем внести напечатанные `RISE`/`FALL` в `src/autolabel.py` и перезапустить `python -m src.autolabel`.

- [ ] **Step 6: Коммит**

```bash
git add -A
git commit -m "feat: блочное разбиение выборки и калибровка коэффициентов разметки"
```

---

### Task 9: Классический baseline

**Files:**
- Create: `src/baseline.py`
- Test: `tests/test_baseline.py`

**Interfaces:**
- Consumes: `audio.*`, `autolabel.interval_from_audio`
- Produces: `baseline.predict(path) -> dict` с ключами `start`, `end`, `confidence`

Это полноправная точка сравнения, а не заглушка. Если модель её не обыграет, об этом надо сказать прямо.

- [ ] **Step 1: Написать падающий тест**

```python
# tests/test_baseline.py
import numpy as np, pytest
from src import baseline, config
from tests.test_audio_features import _pulse_train

def test_baseline_finds_beats_in_the_middle(tmp_path):
    """Удары только с 8-й по 18-ю секунду на фоне шума."""
    sr = config.SR
    x = np.random.default_rng(0).normal(0, 0.02, sr * 26).astype(np.float32)
    beats = _pulse_train(26.0, 70, noise=0.0, start_s=8.0, end_s=18.0)
    x = x + beats
    r = baseline.predict_array(x)
    assert abs(r["start"] - 8.0) < 2.5
    assert abs(r["end"] - 18.0) < 2.5

def test_baseline_returns_none_interval_on_pure_noise():
    x = np.random.default_rng(1).normal(0, 0.1, config.SR * 20).astype(np.float32)
    r = baseline.predict_array(x)
    assert r["confidence"] < 0.5
```

- [ ] **Step 2: Убедиться, что тест падает**

Run: `python -m pytest tests/test_baseline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.baseline'`

- [ ] **Step 3: Реализовать**

```python
# src/baseline.py
import json, sys
import numpy as np
from . import audio, autolabel, config

def predict_array(x):
    iv = autolabel.interval_from_audio(x)
    env = audio.rms_envelope(audio.bandpass(x, config.BAND_LO, config.BAND_HI))
    per = audio.periodicity(env)
    if iv is None:
        return {"start": 0.0, "end": float(len(x) / config.SR), "confidence": 0.0}
    a = int(iv[0] / config.HOP_S)
    b = max(a + 1, int(iv[1] / config.HOP_S))
    return {"start": float(iv[0]), "end": float(iv[1]),
            "confidence": float(np.mean(per[a:b])) if b <= len(per) else 0.0}

def predict(path):
    x, _ = audio.load_8k(path)
    return predict_array(x)

if __name__ == "__main__":
    print(json.dumps(predict(sys.argv[1]), ensure_ascii=False))
```

- [ ] **Step 4: Убедиться, что тесты проходят**

Run: `python -m pytest tests/test_baseline.py -v`
Expected: PASS, 2 теста

- [ ] **Step 5: Коммит**

```bash
git add -A
git commit -m "feat: классический детектор как точка сравнения"
```

---

### Task 10: Dataset и аугментации

**Files:**
- Create: `src/dataset.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `audio.features`, `splits.load`, CSV меток
- Produces:
  - `dataset.interval_to_mask(start, end, n_frames) -> np.ndarray` — float32 (T,)
  - `dataset.mask_to_interval(mask, threshold=0.5) -> tuple[float,float] | None`
  - `dataset.KorotkoffDataset(ids, labels, augment=False)` — `__getitem__` даёт `(features, mask, id)`
  - `dataset.collate(batch) -> tuple[Tensor, Tensor, Tensor]` — `(x, y, lengths)` с паддингом

- [ ] **Step 1: Написать падающие тесты**

```python
# tests/test_dataset.py
import numpy as np, torch
from src import dataset, config

def test_interval_to_mask_marks_the_right_frames():
    m = dataset.interval_to_mask(2.0, 4.0, n_frames=500)   # 10 с при шаге 20 мс
    assert m.shape == (500,)
    assert m[:100].sum() == 0
    assert m[100:200].min() == 1.0
    assert m[200:].sum() == 0

def test_mask_to_interval_is_inverse_of_interval_to_mask():
    m = dataset.interval_to_mask(3.0, 7.5, n_frames=600)
    s, e = dataset.mask_to_interval(m)
    assert abs(s - 3.0) < config.HOP_S * 1.5
    assert abs(e - 7.5) < config.HOP_S * 1.5

def test_mask_to_interval_returns_none_when_empty():
    assert dataset.mask_to_interval(np.zeros(100, dtype=np.float32)) is None

def test_collate_pads_and_reports_lengths():
    a = (torch.randn(100, config.N_FEATURES), torch.rand(100), 1)
    b = (torch.randn(60, config.N_FEATURES), torch.rand(60), 2)
    x, y, lens = dataset.collate([a, b])
    assert x.shape == (2, 100, config.N_FEATURES)
    assert y.shape == (2, 100)
    assert lens.tolist() == [100, 60]
    assert torch.all(x[1, 60:] == 0)
```

- [ ] **Step 2: Убедиться, что тесты падают**

Run: `python -m pytest tests/test_dataset.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.dataset'`

- [ ] **Step 3: Реализовать**

```python
# src/dataset.py
import csv
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import signal
from . import audio, config, manifest

def interval_to_mask(start, end, n_frames):
    m = np.zeros(n_frames, dtype=np.float32)
    a = int(np.clip(round(start / config.HOP_S), 0, n_frames))
    b = int(np.clip(round(end / config.HOP_S), 0, n_frames))
    m[a:b] = 1.0
    return m

def mask_to_interval(mask, threshold=0.5):
    idx = np.where(np.asarray(mask) > threshold)[0]
    if len(idx) == 0:
        return None
    return float(idx[0] * config.HOP_S), float((idx[-1] + 1) * config.HOP_S)

def read_labels(path):
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            out[int(row["id"])] = (float(row["start"]), float(row["end"]))
    return out

class KorotkoffDataset(Dataset):
    def __init__(self, ids, labels, augment=False, cache=True):
        self.rows = {r["id"]: r for r in manifest.read()}
        self.ids = [i for i in ids if i in labels and i in self.rows]
        self.labels = labels
        self.augment = augment
        self.cache = cache
        self.rng = np.random.default_rng(0)
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.ids)

    def _signal(self, rid):
        x, _ = audio.load_8k(self.rows[rid]["wav_path"])
        return x

    def _features(self, rid, x=None):
        cpath = config.CACHE_DIR / f"{rid}.npy"
        if self.cache and not self.augment and cpath.exists():
            return np.load(cpath)
        f = audio.features(self._signal(rid) if x is None else x)
        if self.cache and not self.augment:
            np.save(cpath, f)
        return f

    def __getitem__(self, i):
        rid = self.ids[i]
        start, end = self.labels[rid]
        if not self.augment:
            f = self._features(rid)
            y = interval_to_mask(start, end, len(f))
            return torch.from_numpy(f), torch.from_numpy(y), rid

        x = self._signal(rid)
        # усиление
        x = x * float(self.rng.uniform(0.5, 2.0))
        # шум, взятый из тихой части этой же записи
        quiet = x[: int(0.5 * config.SR)]
        if len(quiet) and self.rng.random() < 0.5:
            lvl = float(np.std(quiet)) * float(self.rng.uniform(0.5, 2.0))
            x = x + self.rng.normal(0, max(lvl, 1e-6), len(x)).astype(np.float32)
        # растяжение времени
        if self.rng.random() < 0.5:
            k = float(self.rng.uniform(0.9, 1.1))
            up, dn = int(round(1000 * k)), 1000
            x = signal.resample_poly(x, up, dn).astype(np.float32)
            start, end = start * k, end * k
        # сдвиг
        if self.rng.random() < 0.5:
            shift = float(self.rng.uniform(-1.5, 1.5))
            n = int(abs(shift) * config.SR)
            if shift > 0:
                x = np.concatenate([np.zeros(n, dtype=np.float32), x])
            else:
                x = x[n:]
            start, end = start + shift, end + shift

        dur = len(x) / config.SR
        start = float(np.clip(start, 0, dur))
        end = float(np.clip(end, 0, dur))
        f = audio.features(x)
        y = interval_to_mask(start, end, len(f))
        return torch.from_numpy(f), torch.from_numpy(y), rid

def collate(batch):
    lens = torch.tensor([b[0].shape[0] for b in batch], dtype=torch.long)
    T = int(lens.max())
    x = torch.zeros(len(batch), T, config.N_FEATURES, dtype=torch.float32)
    y = torch.zeros(len(batch), T, dtype=torch.float32)
    for i, (f, m, _) in enumerate(batch):
        x[i, : f.shape[0]] = f
        y[i, : m.shape[0]] = m
    return x, y, lens
```

- [ ] **Step 4: Убедиться, что тесты проходят**

Run: `python -m pytest tests/test_dataset.py -v`
Expected: PASS, 4 теста

- [ ] **Step 5: Коммит**

```bash
git add -A
git commit -m "feat: Dataset с аугментациями и паддингом батчей"
```

---

### Task 11: Модель

**Files:**
- Create: `src/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `config.N_FEATURES`
- Produces:
  - `model.KorotkoffNet(n_features=config.N_FEATURES, channels=(64,64,128,128), dilations=(1,4,8,16), gru_hidden=128)`

Рецептивное поле свёрточной части: `1 + 4*(1+4+8+16) = 117` кадров = **2.34 с**.
С dilation 1/2/4/8 вышло бы всего 1.22 с — меньше двух ударов, чего недостаточно
для оценки периодичности.
  - `forward(x: Tensor[B,T,F]) -> Tensor[B,T]` — логиты на кадр
  - `model.count_parameters(m) -> int`
  - `model.pick_device() -> torch.device`

- [ ] **Step 1: Написать падающие тесты**

```python
# tests/test_model.py
import torch
from src import model, config

def test_forward_preserves_time_dimension():
    m = model.KorotkoffNet()
    x = torch.randn(3, 250, config.N_FEATURES)
    y = m(x)
    assert y.shape == (3, 250)

def test_forward_handles_variable_length():
    m = model.KorotkoffNet()
    for T in (60, 137, 1400):
        assert m(torch.randn(1, T, config.N_FEATURES)).shape == (1, T)

def test_parameter_count_in_expected_range():
    n = model.count_parameters(model.KorotkoffNet())
    assert 100_000 < n < 600_000, f"неожиданный размер модели: {n}"

def test_receptive_field_spans_multiple_beats():
    """Изменение одного кадра должно влиять минимум на 2 с вокруг него —
    иначе сеть не увидит периодичность."""
    m = model.KorotkoffNet().eval()
    T = 400
    x = torch.zeros(1, T, config.N_FEATURES, requires_grad=True)
    out = m(x)
    out[0, T // 2].backward()
    influenced = (x.grad[0].abs().sum(dim=1) > 0).nonzero().flatten()
    span = float(influenced.max() - influenced.min()) * config.HOP_S
    assert span > 2.0, f"рецептивное поле всего {span:.2f} с"
```

- [ ] **Step 2: Убедиться, что тесты падают**

Run: `python -m pytest tests/test_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.model'`

- [ ] **Step 3: Реализовать**

```python
# src/model.py
import torch
import torch.nn as nn
from . import config

class KorotkoffNet(nn.Module):
    """Расширенные свёртки набирают контекст в несколько ударов (2.34 с),
    BiGRU связывает всю запись, линейный слой даёт логит на каждый кадр."""

    def __init__(self, n_features=config.N_FEATURES, channels=(64, 64, 128, 128),
                 dilations=(1, 4, 8, 16), gru_hidden=128, dropout=0.1):
        super().__init__()
        blocks, prev = [], n_features
        for ch, d in zip(channels, dilations):
            blocks += [
                nn.Conv1d(prev, ch, kernel_size=5, dilation=d, padding=2 * d),
                nn.BatchNorm1d(ch),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev = ch
        self.encoder = nn.Sequential(*blocks)
        self.gru = nn.GRU(prev, gru_hidden, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * gru_hidden, 1)

    def forward(self, x):
        h = self.encoder(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.gru(h)
        return self.head(h).squeeze(-1)

def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
```

- [ ] **Step 4: Убедиться, что тесты проходят**

Run: `python -m pytest tests/test_model.py -v`
Expected: PASS, 4 теста

- [ ] **Step 5: Коммит**

```bash
git add -A
git commit -m "feat: CRNN для покадровой сегментации"
```

---

### Task 12: Обучение

**Files:**
- Create: `src/train.py`
- Test: `tests/test_train.py`
- Output: `runs/best.pt`

**Interfaces:**
- Consumes: `model.*`, `dataset.*`, `splits.load`
- Produces:
  - `train.masked_bce(logits, target, lengths, pos_weight) -> Tensor`
  - `train.soft_dice(logits, target, lengths) -> Tensor`
  - `train.loss_fn(logits, target, lengths, pos_weight, dice_weight=0.5) -> Tensor`
  - `train.run(epochs=60, batch_size=8, lr=3e-4) -> Path` — путь к лучшим весам

- [ ] **Step 1: Написать падающие тесты**

```python
# tests/test_train.py
import torch
from src import train

def test_masked_bce_ignores_padding():
    """Мусор в области паддинга не должен влиять на значение потерь."""
    logits = torch.zeros(2, 10)
    target = torch.zeros(2, 10)
    lengths = torch.tensor([10, 4])
    base = train.masked_bce(logits, target, lengths, pos_weight=torch.tensor(1.0))
    logits2 = logits.clone()
    logits2[1, 4:] = 50.0
    assert torch.allclose(base, train.masked_bce(logits2, target, lengths,
                                                 pos_weight=torch.tensor(1.0)))

def test_soft_dice_is_low_for_perfect_prediction():
    target = torch.zeros(1, 100); target[0, 20:60] = 1.0
    logits = torch.where(target > 0, torch.tensor(8.0), torch.tensor(-8.0))
    d = train.soft_dice(logits, target, torch.tensor([100]))
    assert float(d) < 0.05

def test_soft_dice_is_high_for_inverted_prediction():
    target = torch.zeros(1, 100); target[0, 20:60] = 1.0
    logits = torch.where(target > 0, torch.tensor(-8.0), torch.tensor(8.0))
    assert float(train.soft_dice(logits, target, torch.tensor([100]))) > 0.9

def test_loss_decreases_on_a_tiny_overfit_run():
    """Сеть обязана переобучиться на одном примере — иначе цикл обучения сломан."""
    from src import model, config
    m = model.KorotkoffNet()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    x = torch.randn(1, 120, config.N_FEATURES)
    y = torch.zeros(1, 120); y[0, 30:80] = 1.0
    lens = torch.tensor([120])
    pw = torch.tensor(1.0)
    first = None
    for _ in range(40):
        opt.zero_grad()
        l = train.loss_fn(m(x), y, lens, pw)
        l.backward(); opt.step()
        first = float(l) if first is None else first
    assert float(l) < first * 0.5
```

- [ ] **Step 2: Убедиться, что тесты падают**

Run: `python -m pytest tests/test_train.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.train'`

- [ ] **Step 3: Реализовать**

```python
# src/train.py
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from . import config, dataset, model, splits

def _mask_from_lengths(lengths, T, device):
    ar = torch.arange(T, device=device)[None, :]
    return (ar < lengths[:, None].to(device)).float()

def masked_bce(logits, target, lengths, pos_weight):
    m = _mask_from_lengths(lengths, logits.shape[1], logits.device)
    loss = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none",
        pos_weight=pos_weight.to(logits.device))
    return (loss * m).sum() / m.sum().clamp(min=1.0)

def soft_dice(logits, target, lengths, eps=1e-6):
    m = _mask_from_lengths(lengths, logits.shape[1], logits.device)
    p = torch.sigmoid(logits) * m
    t = target * m
    inter = (p * t).sum(dim=1)
    denom = p.sum(dim=1) + t.sum(dim=1)
    return (1.0 - (2 * inter + eps) / (denom + eps)).mean()

def loss_fn(logits, target, lengths, pos_weight, dice_weight=0.5):
    return masked_bce(logits, target, lengths, pos_weight) + \
           dice_weight * soft_dice(logits, target, lengths)

def _iou(pred, target, lengths):
    m = _mask_from_lengths(lengths, pred.shape[1], pred.device)
    p = ((torch.sigmoid(pred) > 0.5).float() * m)
    t = target * m
    inter = (p * t).sum(dim=1)
    union = ((p + t) > 0).float().mul(m).sum(dim=1)
    return float((inter / union.clamp(min=1.0)).mean())

def run(epochs=60, batch_size=8, lr=3e-4, patience=10, out_dir=None):
    out_dir = Path(out_dir or config.PROJECT_ROOT / "runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    sp = splits.load()
    labels = dataset.read_labels(config.DATA_DIR / "labels_auto.csv")

    tr = dataset.KorotkoffDataset(sp["train"], labels, augment=True)
    va = dataset.KorotkoffDataset(sp["val"], labels, augment=False)
    dl_tr = DataLoader(tr, batch_size=batch_size, shuffle=True,
                       collate_fn=dataset.collate, num_workers=0)
    dl_va = DataLoader(va, batch_size=batch_size, shuffle=False,
                       collate_fn=dataset.collate, num_workers=0)

    # доля положительных кадров задаёт вес класса
    pos = np.mean([(labels[i][1] - labels[i][0]) for i in tr.ids])
    dur = np.mean([tr.rows[i]["dur_s"] for i in tr.ids])
    frac = float(np.clip(pos / max(dur, 1e-6), 0.05, 0.95))
    pos_weight = torch.tensor((1 - frac) / frac)
    print(f"доля положительных кадров ~{frac:.2f}, pos_weight={float(pos_weight):.2f}")

    dev = model.pick_device()
    net = model.KorotkoffNet().to(dev)
    print(f"устройство: {dev}, параметров: {model.count_parameters(net)}")
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_iou, best_epoch, best_path = -1.0, -1, out_dir / "best.pt"
    for ep in range(epochs):
        net.train()
        tot = 0.0
        for x, y, lens in dl_tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            l = loss_fn(net(x), y, lens, pos_weight)
            l.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            tot += float(l) * len(x)
        sched.step()

        net.eval()
        ious, vloss = [], 0.0
        with torch.no_grad():
            for x, y, lens in dl_va:
                x, y = x.to(dev), y.to(dev)
                out = net(x)
                vloss += float(loss_fn(out, y, lens, pos_weight)) * len(x)
                ious.append(_iou(out, y, lens))
        iou = float(np.mean(ious)) if ious else 0.0
        print(f"эпоха {ep:3d}  train {tot/max(len(tr),1):.4f}  "
              f"val {vloss/max(len(va),1):.4f}  IoU {iou:.4f}")

        if iou > best_iou:
            best_iou, best_epoch = iou, ep
            torch.save({"state_dict": net.state_dict(), "iou": iou, "epoch": ep}, best_path)
        elif ep - best_epoch >= patience:
            print(f"ранняя остановка: IoU не растёт {patience} эпох")
            break

    print(f"лучший IoU {best_iou:.4f} на эпохе {best_epoch}, веса в {best_path}")
    return best_path

if __name__ == "__main__":
    run()
```

- [ ] **Step 4: Убедиться, что тесты проходят**

Run: `python -m pytest tests/test_train.py -v`
Expected: PASS, 4 теста. Последний проверяет, что сеть способна переобучиться на одном примере — если он падает, цикл обучения сломан, и дальше идти нельзя.

- [ ] **Step 5: Обучить**

Run: `python -m src.train`
Expected: IoU на валидации растёт и выходит на плато выше 0.75. Если застревает ниже 0.5 — проблема в метках, а не в сети; вернуться к Task 6.

- [ ] **Step 6: Коммит**

```bash
git add -A
git commit -m "feat: цикл обучения с BCE и Dice, ранняя остановка по IoU"
```

---

### Task 13: Постобработка и инференс

**Files:**
- Create: `src/postprocess.py`, `src/predict.py`
- Test: `tests/test_postprocess.py`

**Interfaces:**
- Consumes: `config.*`
- Produces:
  - `postprocess.smooth(prob, win_s=0.5) -> np.ndarray`
  - `postprocess.close_gaps(mask, max_gap_s=3.0) -> np.ndarray`
  - `postprocess.longest_run(mask) -> tuple[int,int] | None`
  - `postprocess.refine_edge(prob, idx, threshold) -> float`
  - `postprocess.to_interval(prob, threshold=0.5, max_gap_s=3.0) -> dict | None`
  - `predict.predict(path, ckpt=None) -> dict` — `{"start","end","confidence"}`

- [ ] **Step 1: Написать падающие тесты**

```python
# tests/test_postprocess.py
import numpy as np
from src import postprocess as pp, config

def test_close_gaps_bridges_short_hole():
    """Аускультативный провал: тоны пропадают на 2 с и возвращаются."""
    m = np.zeros(500, dtype=bool)
    m[100:200] = True
    m[300:400] = True                       # дыра 100 кадров = 2.0 с
    out = pp.close_gaps(m, max_gap_s=3.0)
    assert out[200:300].all()
    assert not out[:100].any() and not out[400:].any()

def test_close_gaps_keeps_long_hole_open():
    m = np.zeros(600, dtype=bool)
    m[50:100] = True
    m[400:450] = True                       # дыра 300 кадров = 6.0 с
    out = pp.close_gaps(m, max_gap_s=3.0)
    assert not out[200:300].any()

def test_longest_run_picks_the_longest():
    m = np.zeros(100, dtype=bool)
    m[5:10] = True
    m[40:70] = True
    assert pp.longest_run(m) == (40, 70)

def test_to_interval_recovers_a_clean_block():
    prob = np.zeros(500, dtype=np.float32)
    prob[150:350] = 0.9
    r = pp.to_interval(prob)
    assert abs(r["start"] - 150 * config.HOP_S) < 0.1
    assert abs(r["end"] - 350 * config.HOP_S) < 0.1
    assert r["confidence"] > 0.8

def test_to_interval_returns_none_when_nothing_crosses_threshold():
    assert pp.to_interval(np.full(300, 0.1, dtype=np.float32)) is None

def test_refine_edge_gives_subframe_precision():
    prob = np.zeros(200, dtype=np.float32)
    prob[100:] = 1.0
    prob[99] = 0.5                           # граница ровно на кадре 99
    t = pp.refine_edge(prob, 100, threshold=0.5)
    assert 99 * config.HOP_S <= t <= 100 * config.HOP_S
```

- [ ] **Step 2: Убедиться, что тесты падают**

Run: `python -m pytest tests/test_postprocess.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.postprocess'`

- [ ] **Step 3: Реализовать `src/postprocess.py`**

```python
# src/postprocess.py
import numpy as np
from scipy import signal
from . import config

def smooth(prob, win_s=0.5):
    k = int(win_s / config.HOP_S) | 1
    if len(prob) < k:
        return np.asarray(prob, dtype=np.float32)
    return signal.medfilt(np.asarray(prob, dtype=np.float64), kernel_size=k).astype(np.float32)

def close_gaps(mask, max_gap_s=3.0):
    """Сшивает короткие разрывы. Аускультативный провал — не конец интервала."""
    m = np.asarray(mask, dtype=bool).copy()
    max_gap = int(max_gap_s / config.HOP_S)
    idx = np.where(m)[0]
    if len(idx) < 2:
        return m
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_gap + 1:
            m[a:b] = True
    return m

def longest_run(mask):
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return None
    padded = np.concatenate([[False], m, [False]])
    d = np.diff(padded.astype(np.int8))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    k = int(np.argmax(ends - starts))
    return int(starts[k]), int(ends[k])

def refine_edge(prob, idx, threshold):
    """Уточняет границу линейной интерполяцией между кадрами idx-1 и idx.

    Формула одна для обоих концов: на подъёме p[idx-1] <= thr < p[idx],
    на спаде p[idx-1] > thr >= p[idx]. В обоих случаях числитель и знаменатель
    меняют знак вместе, поэтому доля получается положительной.
    """
    p = np.asarray(prob, dtype=np.float64)
    i = int(np.clip(idx, 1, len(p) - 1))
    a, b = p[i - 1], p[i]
    if abs(b - a) < 1e-9:
        return i * config.HOP_S
    frac = float(np.clip((threshold - a) / (b - a), 0.0, 1.0))
    return (i - 1 + frac) * config.HOP_S

def to_interval(prob, threshold=0.5, max_gap_s=3.0):
    p = smooth(prob)
    m = close_gaps(p > threshold, max_gap_s)
    run = longest_run(m)
    if run is None:
        return None
    a, b = run
    start = refine_edge(p, a, threshold) if a > 0 else 0.0
    end = refine_edge(p, b, threshold) if b < len(p) else len(p) * config.HOP_S
    if end <= start:
        end = start + config.HOP_S
    return {"start": float(start), "end": float(end),
            "confidence": float(np.mean(p[a:b])) if b > a else 0.0}
```

- [ ] **Step 4: Реализовать `src/predict.py`**

```python
# src/predict.py
import argparse, json
from pathlib import Path
import numpy as np
import torch
from . import audio, config, model, postprocess

_CACHE = {}

def load_model(ckpt=None):
    ckpt = Path(ckpt or config.PROJECT_ROOT / "runs" / "best.pt")
    key = str(ckpt)
    if key not in _CACHE:
        dev = model.pick_device()
        net = model.KorotkoffNet().to(dev)
        net.load_state_dict(torch.load(ckpt, map_location=dev)["state_dict"])
        net.eval()
        _CACHE[key] = (net, dev)
    return _CACHE[key]

def frame_probabilities(path, ckpt=None):
    net, dev = load_model(ckpt)
    x, _ = audio.load_8k(path)
    f = torch.from_numpy(audio.features(x))[None].to(dev)
    with torch.no_grad():
        return torch.sigmoid(net(f))[0].cpu().numpy(), len(x) / config.SR

def predict(path, ckpt=None, threshold=0.5):
    prob, dur = frame_probabilities(path, ckpt)
    r = postprocess.to_interval(prob, threshold=threshold)
    if r is None:
        return {"start": None, "end": None, "confidence": 0.0, "duration": dur}
    r["start"] = float(np.clip(r["start"], 0.0, dur))
    r["end"] = float(np.clip(r["end"], 0.0, dur))
    r["duration"] = dur
    return r

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Интервал тонов Короткова в WAV")
    ap.add_argument("wav")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()
    print(json.dumps(predict(a.wav, a.ckpt, a.threshold), ensure_ascii=False, indent=2))
```

- [ ] **Step 5: Убедиться, что тесты проходят**

Run: `python -m pytest tests/test_postprocess.py -v`
Expected: PASS, 6 тестов

- [ ] **Step 6: Проверить инференс**

Run: `python -m src.predict "$PIVOT_DATA_ROOT/recording/recording (100).wav"`
Expected: JSON с интервалом, пересекающимся с 6–17 с.

- [ ] **Step 7: Коммит**

```bash
git add -A
git commit -m "feat: постобработка кривой вероятностей и CLI инференса"
```

---

### Task 14: Оценка и отчёт

**Files:**
- Create: `src/evaluate.py`
- Test: `tests/test_evaluate.py`
- Output: `reports/evaluation.md`

**Interfaces:**
- Consumes: `predict.*`, `baseline.*`, `pressure.*`, `splits.load`
- Produces:
  - `evaluate.boundary_errors(pred, truth) -> dict` — `start_err`, `end_err`, `iou`
  - `evaluate.to_mmhg(rid, t_start, t_end) -> tuple[float,float] | None`
  - `evaluate.summarise(results) -> dict`
  - `evaluate.report(model_res, baseline_res) -> str`

- [ ] **Step 1: Написать падающие тесты**

```python
# tests/test_evaluate.py
import numpy as np
from src import evaluate as ev

def test_boundary_errors_are_signed_free_and_iou_correct():
    r = ev.boundary_errors((5.0, 15.0), (4.5, 15.5))
    assert abs(r["start_err"] - 0.5) < 1e-9
    assert abs(r["end_err"] - 0.5) < 1e-9
    assert abs(r["iou"] - 10.0 / 11.0) < 1e-6

def test_iou_is_zero_for_disjoint_intervals():
    assert ev.boundary_errors((1.0, 2.0), (10.0, 12.0))["iou"] == 0.0

def test_summarise_reports_hit_rates():
    res = [{"start_err": 0.2, "end_err": 0.3, "iou": 0.9},
           {"start_err": 1.4, "end_err": 0.4, "iou": 0.7},
           {"start_err": 0.1, "end_err": 0.1, "iou": 0.95}]
    s = ev.summarise(res)
    assert abs(s["start_mae"] - (0.2 + 1.4 + 0.1) / 3) < 1e-9
    assert abs(s["start_within_0.5"] - 2 / 3) < 1e-9
    assert abs(s["start_within_1.0"] - 2 / 3) < 1e-9
    assert abs(s["median_iou"] - 0.9) < 1e-9

def test_summarise_handles_empty_input():
    s = ev.summarise([])
    assert s["n"] == 0
```

- [ ] **Step 2: Убедиться, что тесты падают**

Run: `python -m pytest tests/test_evaluate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.evaluate'`

- [ ] **Step 3: Реализовать**

```python
# src/evaluate.py
import numpy as np
from . import config, manifest, pressure, predict, baseline, splits

def boundary_errors(pred, truth):
    ps, pe = pred
    ts, te = truth
    inter = max(0.0, min(pe, te) - max(ps, ts))
    union = max(pe, te) - min(ps, ts)
    return {"start_err": abs(ps - ts), "end_err": abs(pe - te),
            "iou": float(inter / union) if union > 0 else 0.0}

_ROWS = None

def to_mmhg(rid, t_start, t_end):
    """Пересчитывает границы в мм рт. ст. по синхронной кривой давления."""
    global _ROWS
    if _ROWS is None:
        _ROWS = {r["id"]: r for r in manifest.read()}
    r = _ROWS.get(rid)
    if not r or not r["has_pressure"] or not r["pressure_path"]:
        return None
    t, p = pressure.read_pressure(r["pressure_path"])
    tu, pu = pressure.resample_uniform(t, p)
    off = config.SYNC_OFFSET_S
    return (pressure.pressure_at(tu, pu, t_start + off),
            pressure.pressure_at(tu, pu, t_end + off))

def summarise(results):
    if not results:
        return {"n": 0}
    se = np.array([r["start_err"] for r in results])
    ee = np.array([r["end_err"] for r in results])
    iou = np.array([r["iou"] for r in results])
    out = {"n": len(results),
           "start_mae": float(se.mean()), "start_median": float(np.median(se)),
           "end_mae": float(ee.mean()), "end_median": float(np.median(ee)),
           "median_iou": float(np.median(iou)),
           "start_within_0.5": float(np.mean(se <= 0.5)),
           "start_within_1.0": float(np.mean(se <= 1.0)),
           "end_within_0.5": float(np.mean(ee <= 0.5)),
           "end_within_1.0": float(np.mean(ee <= 1.0))}
    sb = np.array([r["sbp_err"] for r in results if "sbp_err" in r])
    db = np.array([r["dbp_err"] for r in results if "dbp_err" in r])
    if len(sb):
        out.update(sbp_bias=float(sb.mean()), sbp_sd=float(sb.std()),
                   dbp_bias=float(db.mean()), dbp_sd=float(db.std()))
    return out

def evaluate_on(ids, truth, predictor):
    out = []
    rows = {r["id"]: r for r in manifest.read()}
    for rid in ids:
        r = rows.get(rid)
        if not r:
            continue
        try:
            p = predictor(r["wav_path"])
        except Exception:
            continue
        if p.get("start") is None:
            continue
        e = boundary_errors((p["start"], p["end"]), truth[rid])
        e["id"] = rid
        mp = to_mmhg(rid, p["start"], p["end"])
        mt = to_mmhg(rid, *truth[rid])
        if mp and mt:
            e["sbp_err"] = mp[0] - mt[0]
            e["dbp_err"] = mp[1] - mt[1]
        out.append(e)
    return out

def _fmt(s):
    if s["n"] == 0:
        return "нет данных"
    lines = [f"- записей: {s['n']}",
             f"- начало: MAE {s['start_mae']:.2f} с, медиана {s['start_median']:.2f} с",
             f"- конец:  MAE {s['end_mae']:.2f} с, медиана {s['end_median']:.2f} с",
             f"- в пределах ±0.5 с: начало {100*s['start_within_0.5']:.0f}%, "
             f"конец {100*s['end_within_0.5']:.0f}%",
             f"- в пределах ±1.0 с: начало {100*s['start_within_1.0']:.0f}%, "
             f"конец {100*s['end_within_1.0']:.0f}%",
             f"- медианный IoU: {s['median_iou']:.3f}"]
    if "sbp_bias" in s:
        lines += [f"- систолическое: смещение {s['sbp_bias']:+.1f}, СКО {s['sbp_sd']:.1f} мм рт. ст.",
                  f"- диастолическое: смещение {s['dbp_bias']:+.1f}, СКО {s['dbp_sd']:.1f} мм рт. ст."]
    return "\n".join(lines)

def report(model_s, base_s):
    aami = "не оценено"
    if "sbp_bias" in model_s:
        ok = (abs(model_s["sbp_bias"]) <= 5 and model_s["sbp_sd"] <= 8 and
              abs(model_s["dbp_bias"]) <= 5 and model_s["dbp_sd"] <= 8)
        aami = "соответствует" if ok else "НЕ соответствует"
    verdict = "не определено"
    if model_s["n"] and base_s["n"]:
        verdict = ("модель лучше" if model_s["start_mae"] + model_s["end_mae"] <
                   base_s["start_mae"] + base_s["end_mae"] else
                   "**классический детектор не хуже модели**")
    return "\n".join([
        "# Оценка на проверенном вручную тестовом наборе", "",
        "Тестовые файлы не участвовали ни в обучении, ни в калибровке коэффициентов, "
        "ни в подборе порога.", "",
        "## Модель", "", _fmt(model_s), "",
        "## Классический baseline", "", _fmt(base_s), "",
        f"## Итог: {verdict}", "",
        f"Соответствие AAMI (смещение ≤5, СКО ≤8 мм рт. ст.): **{aami}**", ""])

if __name__ == "__main__":
    sp = splits.load()
    truth = splits.read_verified()
    ids = [i for i in sp["verified_test"] if i in truth]
    m = summarise(evaluate_on(ids, truth, lambda p: predict.predict(p)))
    b = summarise(evaluate_on(ids, truth, baseline.predict))
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    text = report(m, b)
    (config.REPORT_DIR / "evaluation.md").write_text(text)
    print(text)
```

- [ ] **Step 4: Убедиться, что тесты проходят**

Run: `python -m pytest tests/test_evaluate.py -v`
Expected: PASS, 4 теста

- [ ] **Step 5: Оценить число уникальных испытуемых**

Спека называет это риском: если записей много, а людей мало, эффективный размер
выборки меньше 906 и любая оценка завышена. Прокси-признак — резкие скачки в
характеристиках соседних по номеру записей.

```python
# добавить в src/evaluate.py
def estimate_subject_blocks(threshold=2.5):
    """Грубая оценка числа испытуемых: считает разрывы в ряду характеристик,
    упорядоченном по номеру файла. Это оценка сверху по числу сессий, а не
    точное число людей — интерпретировать как порядок величины."""
    rows = sorted((r for r in manifest.read() if r["status"] == "ok"),
                  key=lambda r: r["id"])
    feats = np.array([[r["dur_s"], np.log10(max(r["peak"], 1.0)),
                       r["deflation_rate"] if r["deflation_rate"] != "" else 0.0]
                      for r in rows], dtype=float)
    z = (feats - feats.mean(0)) / (feats.std(0) + 1e-9)
    jumps = np.linalg.norm(np.diff(z, axis=0), axis=1)
    n_blocks = int(np.sum(jumps > threshold)) + 1
    return {"n_recordings": len(rows), "n_blocks_estimate": n_blocks,
            "recordings_per_block": len(rows) / max(n_blocks, 1)}
```

Дописать вызов в блок `__main__` и в `report()` строкой
«оценка числа сессий: N, записей на сессию: M».

- [ ] **Step 6: Прогнать полный набор тестов**

Run: `python -m pytest -v`
Expected: все зелёные.

- [ ] **Step 7: Построить отчёт**

Run: `python -m src.evaluate`
Expected: `reports/evaluation.md` с обеими колонками. Если строка итога говорит, что классический детектор не хуже — так и сообщить человеку, не переформулируя.

- [ ] **Step 8: Коммит**

```bash
git add -A
git commit -m "feat: оценка в секундах и мм рт. ст. со сравнением против baseline"
```
