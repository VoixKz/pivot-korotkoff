"""Экспорт обученной модели в ONNX для запуска в браузере.

GitHub Pages отдаёт только статику, поэтому модель должна считаться на
стороне клиента. ONNX Runtime Web делает это в WebAssembly.

Ось времени объявлена динамической: записи разной длины, а переэкспортировать
модель под каждую длину нельзя.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import audio, config, model  # noqa: E402


def export(ckpt_path, out_path, opset=17):
    dev = torch.device("cpu")
    net = model.KorotkoffNet().to(dev)
    state = torch.load(ckpt_path, map_location=dev, weights_only=True)
    net.load_state_dict(state["state_dict"])
    net.eval()

    # BatchNorm в eval-режиме работает по накопленной статистике, поэтому
    # длина примера на экспорт не влияет на результат.
    dummy = torch.randn(1, 600, config.N_FEATURES)
    # dynamo=False — экспортёр на TorchScript. Новый dynamo-путь игнорирует
    # dynamic_axes: он зашил длину примера в Reshape перед линейным слоем,
    # и модель падала на любой другой длине записи.
    torch.onnx.export(
        net, dummy, str(out_path),
        input_names=["features"], output_names=["logits"],
        dynamic_axes={"features": {0: "batch", 1: "time"},
                      "logits": {0: "batch", 1: "time"}},
        opset_version=opset,
        dynamo=False,
    )
    return net, state


def check_parity(net, onnx_path, lengths=(300, 977, 1900), tol=2e-4):
    """ONNX должен давать то же, что PyTorch, на длинах, отличных от экспортной."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    worst = 0.0
    rng = np.random.default_rng(0)
    for T in lengths:
        x = rng.standard_normal((1, T, config.N_FEATURES)).astype(np.float32)
        with torch.no_grad():
            ref = net(torch.from_numpy(x)).numpy()
        got = sess.run(["logits"], {"features": x})[0]
        d = float(np.max(np.abs(ref - got)))
        worst = max(worst, d)
        print(f"  T={T:5d}  максимальное расхождение {d:.2e}  {'ок' if d < tol else 'ПРЕВЫШЕНО'}")
    return worst


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Экспорт модели в ONNX")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    ckpt = Path(a.ckpt or config.PROJECT_ROOT / "runs" / "best.pt")
    out = Path(a.out or config.PROJECT_ROOT / "web" / "model.onnx")
    out.parent.mkdir(parents=True, exist_ok=True)

    net, state = export(ckpt, out)
    print(f"экспортировано: {out}  ({out.stat().st_size / 1e6:.2f} МБ)")
    print(f"  чекпойнт: эпоха {state['epoch']}, IoU {state['iou']:.4f}")
    print("сверка с PyTorch:")
    worst = check_parity(net, out)
    print(f"худшее расхождение {worst:.2e}")

    meta = {
        "sr": config.SR, "hop_s": config.HOP_S, "win_s": config.WIN_S,
        "n_bands": config.N_BANDS, "n_features": config.N_FEATURES,
        "band_lo": config.BAND_LO, "band_hi": config.BAND_HI,
        "mel_lo": config.MEL_LO, "mel_hi": config.MEL_HI,
        "ibi_min_s": config.IBI_MIN_S, "ibi_max_s": config.IBI_MAX_S,
        "periodicity_win_s": config.PERIODICITY_WIN_S,
        "iou": float(state["iou"]), "epoch": int(state["epoch"]),
        "n_train": int(state["n_train"]),
    }
    (out.parent / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"метаданные: {out.parent / 'model_meta.json'}")
