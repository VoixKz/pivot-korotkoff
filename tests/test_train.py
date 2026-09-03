import torch

from src import config, model, train


def test_masked_bce_ignores_padding():
    """Мусор в области паддинга не должен влиять на значение потерь."""
    logits = torch.zeros(2, 10)
    target = torch.zeros(2, 10)
    lengths = torch.tensor([10, 4])
    pw = torch.tensor(1.0)
    base = train.masked_bce(logits, target, lengths, pw)
    noisy = logits.clone()
    noisy[1, 4:] = 50.0
    assert torch.allclose(base, train.masked_bce(noisy, target, lengths, pw))


def test_soft_dice_is_low_for_perfect_prediction():
    target = torch.zeros(1, 100)
    target[0, 20:60] = 1.0
    logits = torch.where(target > 0, torch.tensor(8.0), torch.tensor(-8.0))
    assert float(train.soft_dice(logits, target, torch.tensor([100]))) < 0.05


def test_soft_dice_is_high_for_inverted_prediction():
    target = torch.zeros(1, 100)
    target[0, 20:60] = 1.0
    logits = torch.where(target > 0, torch.tensor(-8.0), torch.tensor(8.0))
    assert float(train.soft_dice(logits, target, torch.tensor([100]))) > 0.9


def test_iou_matches_hand_computed_value():
    target = torch.zeros(1, 100)
    target[0, 20:60] = 1.0
    logits = torch.full((1, 100), -8.0)
    logits[0, 40:80] = 8.0
    got = train.batch_iou(logits, target, torch.tensor([100]))
    assert abs(got - 20 / 60) < 1e-4


def test_loss_decreases_on_a_tiny_overfit_run():
    """Сеть обязана переобучиться на одном примере — иначе цикл сломан."""
    torch.manual_seed(0)
    m = model.KorotkoffNet()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    x = torch.randn(1, 120, config.N_FEATURES)
    y = torch.zeros(1, 120)
    y[0, 30:80] = 1.0
    lens = torch.tensor([120])
    pw = torch.tensor(1.0)
    first, last = None, None
    for _ in range(40):
        opt.zero_grad()
        loss = train.loss_fn(m(x), y, lens, pw)
        loss.backward()
        opt.step()
        last = float(loss.detach())
        if first is None:
            first = last
    assert last < first * 0.5, f"потери упали лишь с {first:.3f} до {last:.3f}"
