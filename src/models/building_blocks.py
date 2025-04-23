import torch
import torch.nn as nn
import torch.nn.functional as F # Не используется напрямую, но может пригодиться
from typing import Tuple

class SELayer(nn.Module):
    """
    Squeeze-and-Excitation Layer.
    Адаптивно перевзвешивает каналы признаков.
    """
    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        # Уменьшаем reduction, если каналов меньше, чтобы избежать ошибки
        actual_reduction = max(1, reduction if channel > reduction else channel // 2)

        self.avg_pool = nn.AdaptiveAvgPool2d(1) # Squeeze -> (B, C, 1, 1)
        self.fc = nn.Sequential(
            # Используем Conv2d 1x1 вместо Linear
            nn.Conv2d(channel, channel // actual_reduction, kernel_size=1, bias=False),
            nn.GELU(), # Активация
            nn.Conv2d(channel // actual_reduction, channel, kernel_size=1, bias=False),
            nn.Sigmoid() # Excitation -> веса каналов [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x)       # (B, C, 1, 1)
        y = self.fc(y)             # (B, C, 1, 1) - Веса
        return x * y.expand_as(x)  # Масштабируем


class ResBlock(nn.Module):
    """
    Residual Block с опциональным SE Layer и активацией GELU.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Tuple[int, int], stride: Tuple[int, int] = (1, 1),
                 use_se: bool = True, se_reduction_ratio: int = 16):
        super().__init__()
        # Рассчитываем паддинг для 'same' свертки
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        # Основной путь F(x)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.activation1 = nn.GELU()
        # Вторая свертка всегда с stride=1
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # SE Block (после второй BN, перед сложением)
        self.se = SELayer(out_channels, se_reduction_ratio) if use_se else nn.Identity()

        # Shortcut path (shortcut(x))
        self.shortcut = nn.Sequential() # Identity по умолчанию
        if stride != (1, 1) or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        # Финальная активация (после сложения)
        self.final_activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        out += identity
        out = self.final_activation(out)
        return out