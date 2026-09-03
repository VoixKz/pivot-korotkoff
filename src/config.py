"""Константы конвейера. Единственный источник истины для всех чисел."""
import os
from pathlib import Path

# --- обработка аудио ---
SR = 8000
HOP_S = 0.020
WIN_S = 0.050
N_BANDS = 40
N_FEATURES = N_BANDS + 2

BAND_LO, BAND_HI = 20.0, 200.0
MEL_LO, MEL_HI = 20.0, 500.0

# --- физиология ---
IBI_MIN_S, IBI_MAX_S = 0.33, 1.6  # 37..180 ударов в минуту
PERIODICITY_WIN_S = 4.0

# --- канал давления ---
PRESSURE_SKIP_S = 2.0  # переходный процесс сброса клапана
PRESSURE_FS = 50.0
PRESSURE_BAND = (0.7, 6.0)

SYNC_OFFSET_S = 0.0  # заполняется по результату Task 5: t_давления = t_аудио + offset

# --- исключённые записи, обоснование в спеке ---
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
