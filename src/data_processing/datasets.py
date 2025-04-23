import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
from pathlib import Path
import random
import logging
from typing import Optional, Tuple, List, Dict, Any
import audiomentations # Для type hinting

# --- Импорты из других модулей src ---
try:
    from src.features.spectral import get_features
    from src.utils.path_utils import create_full_path
    from src.data_processing.augmentations import SpecAugmentTransform
except ImportError as e:
    print(f"КРИТИЧЕСКАЯ ОШИБКА ИМПОРТА в datasets.py: {e}"); raise


logger = logging.getLogger(__name__)

class MorseDataset(Dataset):
    """
    Класс Dataset для азбуки Морзе (v10.5 - с мастер-флагом аугментаций).
    """
    def __init__(
        self,
        df: pd.DataFrame,
        char_to_int: Dict[str, int],
        config: Dict[str, Any],
        is_train: bool,
        audio_augmenter: Optional[audiomentations.Compose],
        spec_augment_transform: Optional[SpecAugmentTransform],
        available_map_indices: Optional[List[int]] = None,
        project_root: Optional[Path] = None,
        apply_all_augmentations_flag: bool = False # По умолчанию выключено
    ):
        if df is None or df.empty: raise ValueError("DataFrame не может быть пустым.")
        if char_to_int is None: raise ValueError("Словарь char_to_int не может быть None.")

        self.df = df.copy().reset_index(drop=True)
        self.char_to_int = char_to_int
        self.config = config
        self.is_train = is_train
        self.project_root = project_root if project_root else Path(".")

        # --- Ключевые параметры ---
        self.text_column = config.get('morse_code_column', 'message')
        self.file_id_column = config.get('train_file_column' if is_train else 'test_file_column', 'id')
        self.audio_config = config.get("audio", {})
        self.blank_idx = config.get("ctc", {}).get("blank_idx", 0)
        self.pad_idx = config.get("ctc", {}).get("pad_idx", 0)

        # --- Пути ---
        paths_cfg = config.get("paths", {})
        self.audio_folder_path = self.project_root / paths_cfg.get("data_dir", "data") / 'raw' / paths_cfg.get("audio_folder_name", "morse_dataset/morse_dataset")
        perlin_dir_rel = paths_cfg.get("generic_perlin_maps_dir")
        self.generic_maps_dir = (self.project_root / perlin_dir_rel) if perlin_dir_rel else None

        # --- Настройка аугментаций (с учетом мастер-флага) ---
        # Аудио-аугментатор передается уже готовым (или None)
        self.audio_augmenter = audio_augmenter if is_train and apply_all_augmentations_flag else None
        self.audio_aug_prob = config.get("audio_augmentation", {}).get("p", 1.0) if self.audio_augmenter else 0.0

        # SpecAugment передается уже готовым (или None)
        self.spec_augment_transform = spec_augment_transform if is_train and apply_all_augmentations_flag else None

        # Perlin Noise: включаем только если is_train, мастер-флаг True, карты есть, И флаг в конфиге True
        self.apply_perlin = (
            is_train and
            apply_all_augmentations_flag and # <<< УЧЕТ МАСТЕР-ФЛАГА
            available_map_indices and
            self.generic_maps_dir and
            config.get("perlin_augmentation", {}).get("apply", False)
        )
        self.available_map_indices = available_map_indices if self.apply_perlin else []
        self.perlin_apply_prob = config.get("perlin_augmentation", {}).get("p", 0.0) if self.apply_perlin else 0.0

        # Additive Gaussian: включаем только если is_train, мастер-флаг True, И флаг в конфиге True
        self.apply_additive_gaussian = (
            is_train and
            apply_all_augmentations_flag and # <<< УЧЕТ МАСТЕР-ФЛАГА
            config.get("additive_gaussian_augmentation", {}).get("apply", False)
        )
        self.add_gauss_apply_prob = config.get("additive_gaussian_augmentation", {}).get("p", 0.0) if self.apply_additive_gaussian else 0.0

        logger.info(f"Создан MorseDataset ({'Train' if is_train else 'Val/Test'}): {len(self.df)}. Мастер-аугментации: {apply_all_augmentations_flag}")
        if is_train and apply_all_augmentations_flag:
             logger.info(f"  Активные аугментации: Audio={bool(self.audio_augmenter)}, Perlin={self.apply_perlin}, AddGauss={self.apply_additive_gaussian}, SpecAug={bool(self.spec_augment_transform)}")


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Optional[Tuple[torch.Tensor, torch.Tensor, int, int]]:
        """Загружает данные для одного сэмпла."""
        if not (0 <= idx < len(self.df)): return None
        row = self.df.iloc[idx]
        file_id = row.get(self.file_id_column, f"MISSING_ID_{idx}")
        morse_text = str(row.get(self.text_column, ""))

        try:
            audio_path = create_full_path(file_id, self.audio_folder_path)

            # 1. Извлечение признаков (и аудио-аугментация, если self.audio_augmenter не None)
            features_np = get_features(
                audio_path, self.audio_config,
                augmenter=self.audio_augmenter, # Передается None, если аугментации выключены
                apply_audio_aug_prob=self.audio_aug_prob,
                is_train=self.is_train
            )
            if features_np is None or features_np.size == 0 or not np.isfinite(features_np).all():
                logger.warning(f"Проблемные признаки для {file_id} (idx {idx}). Пропуск.")
                return None

            # 2. Онлайн аугментации спектрограммы (Perlin, Add Gauss) - вызываются только если is_train
            if self.is_train:
                features_np = self._apply_online_augmentations(features_np, file_id)

            # 3. Преобразование в тензор (C=1, T, F)
            features_tensor = torch.from_numpy(features_np.T).float().unsqueeze(0)

            # 4. SpecAugment (применяется только если self.spec_augment_transform не None)
            if self.spec_augment_transform:
                features_tensor = self.spec_augment_transform(features_tensor)

            # Финальная проверка тензора
            if torch.isnan(features_tensor).any() or torch.isinf(features_tensor).any():
                logger.warning(f"NaN/Inf в признаках для {file_id} (idx {idx}) ПОСЛЕ аугментаций. Пропуск.")
                return None

            # 5. Кодирование текста
            encoded_text = torch.tensor(
                [self.char_to_int.get(char, self.blank_idx) for char in morse_text],
                dtype=torch.long
            )

            # 6. Длины
            input_length = features_tensor.shape[1] # T
            target_length = len(encoded_text)
            if input_length <= 0:
                logger.warning(f"Нулевая длина признаков для {file_id} (idx {idx}). Пропуск.")
                return None

            return features_tensor, encoded_text, input_length, target_length

        except Exception as e:
            logger.error(f"Ошибка в __getitem__ для {file_id} (idx {idx}): {e}", exc_info=False)
            return None

    def _apply_online_augmentations(self, features_np: np.ndarray, file_id: str) -> np.ndarray:
        """Применяет Perlin и Additive Gaussian аугментации, если они включены."""
        current_spec = features_np

        # Perlin Noise (проверка флага self.apply_perlin)
        if self.apply_perlin and random.random() < self.perlin_apply_prob:
            current_spec = self._apply_perlin_noise(current_spec, file_id)

        # Additive Gaussian Noise (проверка флага self.apply_additive_gaussian)
        if self.apply_additive_gaussian and random.random() < self.add_gauss_apply_prob:
            try:
                cfg = self.config.get("additive_gaussian_augmentation", {})
                std_dev = random.uniform(cfg.get("std_dev_min", 0.05), cfg.get("std_dev_max", 0.15))
                additive_noise = np.random.normal(0.0, std_dev, current_spec.shape)
                current_spec = np.clip((current_spec + additive_noise.astype(current_spec.dtype)), 0.0, None)
            except Exception as e_add_gauss:
                 logger.debug(f"Ошибка Add Gauss для {file_id}: {e_add_gauss}.")

        return current_spec

    def _apply_perlin_noise(self, features_np: np.ndarray, file_id: str) -> np.ndarray:
        """Применяет шум Перлина (остается отдельным из-за сложности)."""
        if not self.available_map_indices or self.generic_maps_dir is None: return features_np
        try:
            k = random.choice(self.available_map_indices)
            offset_map_path = self.generic_maps_dir / f"offset_map_{k:04d}.npy"
            multi_map_path = self.generic_maps_dir / f"multi_map_{k:04d}.npy"
            if not offset_map_path.exists() or not multi_map_path.exists(): return features_np

            offset_map_full = np.load(offset_map_path); multi_map_full = np.load(multi_map_path)
            F, T_actual = features_np.shape; F_map, T_max = offset_map_full.shape
            if F_map != F or T_max < T_actual: return features_np

            t_start = random.randint(0, T_max - T_actual) if T_max > T_actual else 0
            offset_raw = offset_map_full[:, t_start:t_start + T_actual]
            multi_raw = multi_map_full[:, t_start:t_start + T_actual]

            cfg = self.config.get("perlin_augmentation", {})
            offset_factor = random.uniform(cfg.get("offset_factor_min", 0.05), cfg.get("offset_factor_max", 0.2))
            multi_amplitude = random.uniform(cfg.get("amplitude_min", 0.2), cfg.get("amplitude_max", 0.9))
            chaos_amplitude = random.uniform(cfg.get("chaos_amplitude_min", 0.0), cfg.get("chaos_amplitude_max", 0.10))

            avg_spec_val = max(np.mean(features_np), 1e-6)
            offset_map = np.clip(((offset_raw + 1.0) * 0.5 * (offset_factor * avg_spec_val)), 0.0, None)
            spec_with_offset = np.clip((features_np + offset_map), 0.0, None)

            multiplicative_map = 1.0 + multi_amplitude * multi_raw
            if chaos_amplitude > 1e-6:
                chaos_type = cfg.get("chaos_type", "uniform")
                if chaos_type == 'uniform': random_chaos = np.random.uniform(-chaos_amplitude, chaos_amplitude, multiplicative_map.shape)
                else: random_chaos = np.random.normal(0.0, chaos_amplitude / 3.0, multiplicative_map.shape)
                multiplicative_map += random_chaos.astype(multiplicative_map.dtype)
            multiplicative_map = np.clip(multiplicative_map, 0.0, None)

            return np.clip((spec_with_offset * multiplicative_map), 0.0, None).astype(features_np.dtype)
        except Exception as e_perlin:
            logger.debug(f"Ошибка Perlin для {file_id}: {e_perlin}.")
            return features_np

# --- collate_fn остается без изменений ---
def collate_fn(
    batch: List[Optional[Tuple[torch.Tensor, torch.Tensor, int, int]]],
    pad_idx: int
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Обрабатывает батч данных из MorseDataset (v10.5).
    Возвращает: (B, C, T_max, F), (B, L_max), (B,), (B,)
    """
    try:
        valid_batch = [b for b in batch if b is not None]
        if not valid_batch: return None
        features_list, texts_list, input_lengths_list, target_lengths_list = zip(*valid_batch)
    except (ValueError, TypeError):
        logger.error("Ошибка распаковки батча в collate_fn.")
        return None

    try:
        features_permuted = [f.permute(1, 0, 2) for f in features_list] # (T, C, F)
        features_padded = pad_sequence(features_permuted, batch_first=False, padding_value=0.0) # (T_max, B, C, F)
        features_padded = features_padded.permute(1, 2, 0, 3) # (B, C, T_max, F)
    except Exception as e_pad_feat:
        logger.error(f"Ошибка паддинга признаков: {e_pad_feat}")
        return None

    try:
        texts_padded = pad_sequence(texts_list, batch_first=True, padding_value=pad_idx)
    except Exception as e_pad_text:
        logger.error(f"Ошибка паддинга текстов: {e_pad_text}")
        return None

    input_lengths_tensor = torch.tensor(input_lengths_list, dtype=torch.long)
    target_lengths_tensor = torch.tensor(target_lengths_list, dtype=torch.long)

    if not (features_padded.shape[0] == texts_padded.shape[0] == len(input_lengths_list) == len(target_lengths_list)):
         logger.error("Ошибка collate_fn: Несоответствие размеров батча!")
         return None

    return features_padded, texts_padded, input_lengths_tensor, target_lengths_tensor