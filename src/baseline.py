"""Классический детектор без обучения — полноправная точка сравнения.

Если обученная модель его не обыгрывает, это результат, который надо
сообщить прямо, а не спрятать.
"""
import argparse
import json

from . import audio, autolabel, config


def predict_array(x, sr=config.SR):
    dur = len(x) / sr
    iv = autolabel.interval_from_audio(x, sr)
    conf = autolabel.audio_confidence(x, sr)
    if iv is None:
        return {"start": None, "end": None, "confidence": float(conf), "duration": dur}
    start, end = iv
    if end <= start:
        end = min(dur, start + config.HOP_S)
    return {"start": float(start), "end": float(end),
            "confidence": float(conf), "duration": dur}


def predict(path):
    x, _ = audio.load_8k(path)
    return predict_array(x)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Классический детектор интервала")
    ap.add_argument("wav")
    a = ap.parse_args()
    print(json.dumps(predict(a.wav), ensure_ascii=False, indent=2))
