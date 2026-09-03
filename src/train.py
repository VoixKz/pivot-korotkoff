"""Цикл обучения: BCE со взвешиванием классов плюс Dice, ранняя остановка по IoU."""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import config, dataset, model, splits


def _length_mask(lengths, T, device):
    ar = torch.arange(T, device=device)[None, :]
    return (ar < lengths[:, None].to(device)).float()


def masked_bce(logits, target, lengths, pos_weight):
    m = _length_mask(lengths, logits.shape[1], logits.device)
    loss = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none", pos_weight=pos_weight.to(logits.device)
    )
    return (loss * m).sum() / m.sum().clamp(min=1.0)


def soft_dice(logits, target, lengths, eps=1e-6):
    """Штрафует рваные предсказания там, где покадровая BCE их не замечает."""
    m = _length_mask(lengths, logits.shape[1], logits.device)
    p = torch.sigmoid(logits) * m
    t = target * m
    inter = (p * t).sum(dim=1)
    denom = p.sum(dim=1) + t.sum(dim=1)
    return (1.0 - (2 * inter + eps) / (denom + eps)).mean()


def loss_fn(logits, target, lengths, pos_weight, dice_weight=0.5):
    return masked_bce(logits, target, lengths, pos_weight) + dice_weight * soft_dice(
        logits, target, lengths
    )


def batch_iou(logits, target, lengths, threshold=0.5):
    m = _length_mask(lengths, logits.shape[1], logits.device)
    p = ((torch.sigmoid(logits) > threshold).float() * m)
    t = target * m
    inter = (p * t).sum(dim=1)
    union = (((p + t) > 0).float() * m).sum(dim=1)
    return float((inter / union.clamp(min=1.0)).mean())


def positive_fraction(ds):
    """Доля положительных кадров — задаёт вес класса в BCE."""
    tot_pos = tot_all = 0.0
    for rid in ds.ids:
        s, e = ds.labels[rid]
        tot_pos += max(0.0, e - s)
        tot_all += float(ds.rows[rid]["dur_s"])
    if tot_all <= 0:
        return 0.5
    return float(np.clip(tot_pos / tot_all, 0.05, 0.95))


def run(epochs=60, batch_size=8, lr=3e-4, patience=10, out_dir=None, labels_path=None):
    out_dir = Path(out_dir or config.PROJECT_ROOT / "runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    sp = splits.load()
    labels = dataset.read_labels(labels_path or config.DATA_DIR / "labels_auto.csv")

    tr = dataset.KorotkoffDataset(sp["train"], labels, augment=True)
    va = dataset.KorotkoffDataset(sp["val"], labels, augment=False)
    if len(tr) == 0 or len(va) == 0:
        raise RuntimeError(f"пустая выборка: train={len(tr)}, val={len(va)}")

    dl_tr = DataLoader(tr, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate)
    dl_va = DataLoader(va, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate)

    frac = positive_fraction(tr)
    pos_weight = torch.tensor((1 - frac) / frac)
    dev = model.pick_device()
    net = model.KorotkoffNet().to(dev)
    print(f"обучение: {len(tr)} записей, проверка: {len(va)}", flush=True)
    print(f"доля положительных кадров {frac:.2f}, pos_weight {float(pos_weight):.2f}", flush=True)
    print(f"устройство {dev}, параметров {model.count_parameters(net)}", flush=True)

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_iou, best_epoch = -1.0, -1
    best_path = out_dir / "best.pt"
    for ep in range(epochs):
        net.train()
        tot = 0.0
        for x, y, lens in dl_tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = loss_fn(net(x), y, lens, pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            tot += float(loss) * len(x)
        sched.step()

        net.eval()
        ious, vloss = [], 0.0
        with torch.no_grad():
            for x, y, lens in dl_va:
                x, y = x.to(dev), y.to(dev)
                out = net(x)
                vloss += float(loss_fn(out, y, lens, pos_weight)) * len(x)
                ious.append(batch_iou(out, y, lens))
        iou = float(np.mean(ious)) if ious else 0.0
        print(f"эпоха {ep:3d}  train {tot / len(tr):.4f}  "
              f"val {vloss / len(va):.4f}  IoU {iou:.4f}", flush=True)

        if iou > best_iou:
            best_iou, best_epoch = iou, ep
            torch.save({"state_dict": net.state_dict(), "iou": iou, "epoch": ep,
                        "n_train": len(tr)}, best_path)
        elif ep - best_epoch >= patience:
            print(f"ранняя остановка: IoU не растёт {patience} эпох", flush=True)
            break

    print(f"лучший IoU {best_iou:.4f} на эпохе {best_epoch}, веса в {best_path}", flush=True)
    return best_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Обучение модели сегментации")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--labels", default=None, help="CSV с метками для обучения")
    a = ap.parse_args()
    run(epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, labels_path=a.labels)
