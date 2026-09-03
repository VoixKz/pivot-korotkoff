"""CRNN для покадровой сегментации интервала тонов Короткова."""
import torch
import torch.nn as nn

from . import config


class KorotkoffNet(nn.Module):
    """Расширенные свёртки набирают контекст в несколько ударов, BiGRU
    связывает всю запись, линейный слой даёт логит на каждый кадр.

    Рецептивное поле свёрточной части: 1 + 4*(1+4+8+16) = 117 кадров = 2.34 с.
    С dilation 1/2/4/8 вышло бы 1.22 с — меньше двух ударов при пульсе 60,
    и оценить периодичность было бы нечем.
    """

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


def pick_device(prefer=None):
    """Устройство для обучения и инференса.

    По умолчанию CPU, и это не оплошность. Замер на Apple M5 (батч 8,
    длины как в реальных данных):

        MPS, разные длины      1269 мс/шаг
        MPS, постоянная длина   969 мс/шаг
        CPU, разные длины       419 мс/шаг

    Модель маленькая (356 тыс. параметров), а BiGRU последователен по
    времени: на запись в 1400 кадров приходится 1400 мелких запусков ядер,
    и накладные расходы MPS съедают всю выгоду от ускорителя. Разброс длин
    добавляет MPS ещё около 30% на перекомпиляцию под каждую новую форму.

    CUDA выбирается, если есть: там картина другая, ядра запускаются дешевле.
    Перекрыть выбор можно параметром prefer или флагом --device.
    """
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
