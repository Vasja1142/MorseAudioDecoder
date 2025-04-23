import audiomentations
import torch
import torch.nn as nn
import torchaudio # Для SpecAugment
import warnings
from typing import Optional, Dict

def create_audio_augmenter(config: dict) -> Optional[audiomentations.Compose]:
    """Создает конвейер аудио-аугментаций на основе конфига."""
    aug_config = config.get("audio_augmentation", {})
    if not aug_config.get("apply", False):
        return None

    transforms = []
    pipeline_config = aug_config.get("pipeline", [])
    if not pipeline_config:
        return None

    print("\n--- Создание конвейера Аудио-Аугментаций ---")
    for item in pipeline_config:
        name = item.get("name")
        # Используем .get для безопасного доступа к params
        params = item.get("params", {}).copy() # Копируем
        if not name:
            continue
        try:
            # Обработка старых имен параметров для Gain
            if name == "Gain":
                if "min_gain_in_db" in params:
                    params["min_gain_db"] = params.pop("min_gain_in_db")
                if "max_gain_in_db" in params:
                    params["max_gain_db"] = params.pop("max_gain_in_db")

            transform_class = getattr(audiomentations, name)
            transforms.append(transform_class(**params))
            print(f"  + Аудио-Ауг: {name} (p={params.get('p', 1.0)})")
        except AttributeError:
             print(f"Ошибка: Класс аудио-аугментации '{name}' не найден в audiomentations.")
        except Exception as e:
            print(f"Ошибка создания аудио-аугментации {name}: {e}")

    if not transforms:
        return None
    # Общая вероятность применения задается в Dataset, тут p=1.0
    return audiomentations.Compose(transforms=transforms, p=1.0)

# --- Класс SpecAugmentTransform ---
class SpecAugmentTransform(nn.Module):
    """Применяет SpecAugment к тензору спектрограммы."""
    def __init__(self, aug_config: dict):
        super().__init__()
        self.apply_spec_augment = aug_config.get("apply", False)
        if not self.apply_spec_augment:
            self.transform = nn.Identity()
            return

        transforms = []
        num_freq_masks = aug_config.get("num_freq_masks", 0)
        freq_mask_param = aug_config.get("freq_mask_param", 0)
        if num_freq_masks > 0 and freq_mask_param > 0:
            # Важно: torchaudio.transforms ожидает iid_masks внутри FrequencyMasking
            for _ in range(num_freq_masks):
                transforms.append(torchaudio.transforms.FrequencyMasking(
                    freq_mask_param=freq_mask_param,
                    iid_masks=aug_config.get("iid_masks", True) # Передаем сюда
                ))

        num_time_masks = aug_config.get("num_time_masks", 0)
        time_mask_param = aug_config.get("time_mask_param", 0)
        if num_time_masks > 0 and time_mask_param > 0:
             # Важно: torchaudio.transforms ожидает iid_masks внутри TimeMasking
            for _ in range(num_time_masks):
                transforms.append(torchaudio.transforms.TimeMasking(
                    time_mask_param=time_mask_param,
                    p=1.0, # Вероятность применяется ко всему блоку, здесь всегда 1.0
                    iid_masks=aug_config.get("iid_masks", True) # Передаем сюда
                ))

        if transforms:
            self.transform = nn.Sequential(*transforms)
            print(f"  + SpecAugment: FreqMasks={num_freq_masks}x{freq_mask_param}, TimeMasks={num_time_masks}x{time_mask_param}, iid={aug_config.get('iid_masks', True)}")
        else:
            self.transform = nn.Identity()
            print("  - SpecAugment: Не применяется (параметры не заданы).")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.transform, nn.Identity) or not self.apply_spec_augment or x.numel() == 0: return x
        # Ожидаемый формат входа для Dataset: (C=1, T, F)
        # Для torchaudio нужно (..., F, T)
        if x.dim() == 3: # (C, T, F)
            x_permuted = x.permute(0, 2, 1) # -> (C, F, T)
        elif x.dim() == 4: # (B, C, T, F) - если вдруг используется напрямую
            x_permuted = x.permute(0, 1, 3, 2) # -> (B, C, F, T)
        else:
            warnings.warn(f"SpecAugment: Неожиданная размерность входа {x.shape}. Пропуск.")
            return x

        x_augmented = self.transform(x_permuted)

        # Возвращаем обратно
        if x.dim() == 3:
            x_restored = x_augmented.permute(0, 2, 1) # -> (C, T, F)
        else: # dim == 4
            x_restored = x_augmented.permute(0, 1, 3, 2) # -> (B, C, T, F)
        return x_restored