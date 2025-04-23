import torch
import torch.nn as nn
from typing import Tuple

class SELayer(nn.Module):
    """
    Squeeze-and-Excitation Layer.
    Адаптивно перевзвешивает каналы признаков.
    """
    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        actual_reduction = max(1, reduction if channel > reduction else channel // 2)
        if channel // actual_reduction <= 0: # Доп. проверка
             actual_reduction = channel # Избегаем деления на 0 или отрицательного размера

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // actual_reduction, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(channel // actual_reduction, channel, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y # Используем broadcasting вместо expand_as для эффективности

class ResBlock(nn.Module):
    """
    Residual Block с опциональным SE Layer и активацией GELU.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Tuple[int, int], stride: Tuple[int, int] = (1, 1),
                 use_se: bool = True, se_reduction_ratio: int = 16):
        super().__init__()
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.activation1 = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.se = SELayer(out_channels, se_reduction_ratio) if use_se else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != (1, 1) or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

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