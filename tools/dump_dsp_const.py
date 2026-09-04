"""Коэффициенты фильтров для переноса признаков в браузер.

Вписывать их в dsp.js по памяти нельзя: неверные коэффициенты дают
правдоподобный, но другой фильтр, и модель молча получает не те признаки.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402

if __name__ == "__main__":
    sos = signal.butter(4, [config.BAND_LO, min(config.BAND_HI, config.SR * 0.45)],
                        btype="band", fs=config.SR, output="sos")
    out = {
        "sos": sos.tolist(),
        "zi": signal.sosfilt_zi(sos).tolist(),
        "padlen": int(3 * (2 * sos.shape[0] + 1)),
        "savgol_5_2": signal.savgol_coeffs(5, 2).tolist(),
        "sr": config.SR,
        "hop": int(config.HOP_S * config.SR),
        "win": int(config.WIN_S * config.SR),
        "n_fft": 1 << (int(config.WIN_S * config.SR) - 1).bit_length(),
        "n_bands": config.N_BANDS,
        "mel_lo": config.MEL_LO, "mel_hi": config.MEL_HI,
        "per_win": int(config.PERIODICITY_WIN_S / config.HOP_S),
        "ibi_lo": int(config.IBI_MIN_S / config.HOP_S),
        "ibi_hi": int(config.IBI_MAX_S / config.HOP_S),
    }
    p = config.PROJECT_ROOT / "web" / "dsp_const.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"записано: {p}")
    print("значения SOS и ZI перенести в web/dsp.js вручную и сверить parity.html")
