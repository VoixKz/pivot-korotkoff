import torch

from src import config, model


def test_forward_preserves_time_dimension():
    m = model.KorotkoffNet()
    assert m(torch.randn(3, 250, config.N_FEATURES)).shape == (3, 250)


def test_forward_handles_variable_length():
    m = model.KorotkoffNet()
    for T in (60, 137, 1400):
        assert m(torch.randn(1, T, config.N_FEATURES)).shape == (1, T)


def test_parameter_count_in_expected_range():
    n = model.count_parameters(model.KorotkoffNet())
    assert 100_000 < n < 600_000, f"неожиданный размер модели: {n}"


def test_conv_receptive_field_spans_several_beats():
    """Без BiGRU свёрточная часть обязана видеть больше двух ударов подряд,
    иначе периодичность ей недоступна."""
    m = model.KorotkoffNet().eval()
    T = 600
    x = torch.zeros(1, T, config.N_FEATURES, requires_grad=True)
    h = m.encoder(x.transpose(1, 2)).transpose(1, 2)
    h[0, T // 2].sum().backward()
    touched = (x.grad[0].abs().sum(dim=1) > 0).nonzero().flatten()
    span = float(touched.max() - touched.min() + 1) * config.HOP_S
    assert span > 2.0, f"рецептивное поле свёрток всего {span:.2f} с"


def test_gru_connects_the_whole_recording():
    m = model.KorotkoffNet().eval()
    T = 300
    x = torch.zeros(1, T, config.N_FEATURES, requires_grad=True)
    m(x)[0, T // 2].backward()
    touched = (x.grad[0].abs().sum(dim=1) > 0).nonzero().flatten()
    assert int(touched.min()) == 0 and int(touched.max()) == T - 1


def test_pick_device_returns_a_usable_device():
    d = model.pick_device()
    torch.zeros(2, 2, device=d)
    assert d.type in ("mps", "cuda", "cpu")


def test_pick_device_defaults_away_from_mps():
    """Замер на M5: MPS 1269 мс/шаг против 419 у CPU — BiGRU на нём проигрывает."""
    assert model.pick_device().type != "mps"


def test_pick_device_honours_explicit_preference():
    assert model.pick_device("cpu").type == "cpu"
