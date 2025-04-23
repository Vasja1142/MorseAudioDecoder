import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
from pathlib import Path
import random
import warnings
from typing import Optional, Tuple, List, Dict
import audiomentations # Для type hinting

# --- Импорты из других модулей src ---
from src.features.spectral import get_features
from src.utils.path_utils import create_full_path
from src.data_processing.augmentations import SpecAugmentTransform # Импортируем класс

class MorseDataset(Dataset):
    """
    Класс Dataset для азбуки Морзе (v10.2 - Вынесен в модуль).
    """
    def __init__(
        self,
        df: pd.DataFrame,
        char_to_int: Dict[str, int],
        config: dict, # Оставляем полный конфиг для удобства доступа к путям и параметрам
        is_train: bool,
        # --- Компоненты передаются снаружи ---
        audio_augmenter: Optional[audiomentations.Compose],
        spec_augment_transform: Optional[SpecAugmentTransform],
        available_map_indices: Optional[List[int]] = None,
        project_root: Optional[Path] = None # Добавляем корень проекта для путей
    ):
        self.df = df.copy().reset_index(drop=True)
        self.char_to_int = char_to_int
        self.config = config
        self.is_train = is_train

        # --- Параметры из конфига ---
        self.text_column = config.get('morse_code_column', 'message')
        self.file_id_column = config.get('train_file_column' if is_train else 'test_file_column', 'id')
        self.ctc_config = config.get("ctc", {})
        self.audio_config = config.get("audio", {})
        self.perlin_config = config.get("perlin_augmentation", {}) if is_train else {}
        self.add_gauss_config = config.get("additive_gaussian_augmentation", {}) if is_train else {}

        # --- Флаги и компоненты аугментаций ---
        self.apply_audio_aug = is_train and audio_augmenter is not None and config.get("audio_augmentation", {}).get("apply", False)
        self.apply_perlin = is_train and available_map_indices and self.perlin_config.get("apply", False)
        self.apply_additive_gaussian = is_train and self.add_gauss_config.get("apply", False)
        self.apply_spec_augment = is_train and spec_augment_transform is not None and config.get("spec_augmentation", {}).get("apply", False)

        self.audio_augmenter = audio_augmenter if self.apply_audio_aug else None
        self.audio_aug_prob = config.get("audio_augmentation", {}).get("p", 0.0) if self.apply_audio_aug else 0.0

        self.spec_augment_transform = spec_augment_transform if self.apply_spec_augment else None

        self.available_map_indices = available_map_indices if self.apply_perlin else []
        self.perlin_apply_prob = self.perlin_config.get("p", 0.0) if self.apply_perlin else 0.0
        self.add_gauss_apply_prob = self.add_gauss_config.get("p", 0.0) if self.apply_additive_gaussian else 0.0

        # --- Пути ---
        self.project_root = project_root if project_root else Path(".") # Корень проекта
        self.data_root_str = config.get("paths", {}).get("data_dir", "data")
        self.audio_folder_name = config.get("paths", {}).get("audio_folder_name", "morse_dataset/morse_dataset")
        self.perlin_maps_dir_str = config.get("paths", {}).get("generic_perlin_maps_dir")

        self.raw_data_path = self.project_root / self.data_root_str / 'raw'
        self.audio_folder_path = self.raw_data_path / self.audio_folder_name
        self.generic_maps_dir = (self.project_root / self.perlin_maps_dir_str) if self.apply_perlin and self.perlin_maps_dir_str else None

        # Вывод информации
        mode_str = "Train" if is_train else "Validation/Test"
        print(f"\n--- Создан MorseDataset ({mode_str}) ---")
        print(f"  Количество записей: {len(self.df)}")
        print(f"  Путь к аудио: {self.audio_folder_path}")
        if is_train:
            print("  Аугментации (Train):")
            print(f"    - Аудио: {'ВКЛ' if self.apply_audio_aug else 'ВЫКЛ'} (p={self.audio_aug_prob:.2f})")
            print(f"    - Perlin: {'ВКЛ' if self.apply_perlin else 'ВЫКЛ'} (p={self.perlin_apply_prob:.2f}, Карт: {len(self.available_map_indices)}, Path: {self.generic_maps_dir})")
            print(f"    - Add Gauss: {'ВКЛ' if self.apply_additive_gaussian else 'ВЫКЛ'} (p={self.add_gauss_apply_prob:.2f})")
            print(f"    - SpecAugment: {'ВКЛ' if self.apply_spec_augment else 'ВЫКЛ'}")
        else:
            print("  Аугментации (Validation/Test): ВЫКЛЮЧЕНЫ")
        print("-" * (len(mode_str) + 24))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        if idx >= len(self.df):
            warnings.warn(f"MorseDataset: Индекс {idx} вне диапазона ({len(self.df)}).")
            return None
        row = self.df.iloc[idx]
        file_id = row.get(self.file_id_column, f"unknown_id_at_index_{idx}")
        morse_text = row.get(self.text_column, "")

        try:
            audio_path = create_full_path(file_id, self.audio_folder_path)
        except Exception as e_path:
            print(f"❌ Ошибка формирования пути для ID {file_id} / Индекс {idx}: {e_path}")
            return None

        # 1. Извлечение оригинальной спектрограммы (F, T_actual)
        original_spec_ft = get_features(
            audio_path, self.audio_config,
            augmenter=self.audio_augmenter,
            apply_audio_aug_prob=self.audio_aug_prob,
            is_train=self.is_train
        )
        if original_spec_ft is None: return None
        current_spec_ft = original_spec_ft.copy()
        F, T_actual = current_spec_ft.shape

        # --- Аугментации спектрограммы (только для трейна) ---
        if self.is_train:
            # 2. Perlin Noise
            if self.apply_perlin and random.random() < self.perlin_apply_prob:
                try:
                    k = random.choice(self.available_map_indices)
                    offset_map_path = self.generic_maps_dir / f"offset_map_{k:04d}.npy"
                    multi_map_path = self.generic_maps_dir / f"multi_map_{k:04d}.npy"
                    offset_map_full = np.load(offset_map_path); multi_map_full = np.load(multi_map_path)
                    F_map, T_max = offset_map_full.shape

                    if F_map == F and T_max >= T_actual:
                        if T_max == T_actual: offset_raw, multi_raw = offset_map_full, multi_map_full
                        else: t_start = random.randint(0, T_max - T_actual); offset_raw = offset_map_full[:, t_start:t_start+T_actual]; multi_raw = multi_map_full[:, t_start:t_start+T_actual]

                        offset_factor = random.uniform(self.perlin_config.get("offset_factor_min", 0.05), self.perlin_config.get("offset_factor_max", 0.2))
                        multi_amplitude = random.uniform(self.perlin_config.get("amplitude_min", 0.2), self.perlin_config.get("amplitude_max", 0.9))
                        chaos_amplitude = random.uniform(self.perlin_config.get("chaos_amplitude_min", 0.0), self.perlin_config.get("chaos_amplitude_max", 0.10))

                        avg_spec_val = max(np.mean(current_spec_ft), 1e-6); max_offset = offset_factor * avg_spec_val
                        offset_map = np.clip(((offset_raw + 1.0) * 0.5 * max_offset), 0.0, None)
                        spec_with_offset = np.clip((current_spec_ft + offset_map), 0.0, None)

                        multiplicative_map = 1.0 + multi_amplitude * multi_raw
                        if chaos_amplitude > 1e-6:
                            chaos_type = self.perlin_config.get("chaos_type", "uniform")
                            if chaos_type == 'uniform': random_chaos = np.random.uniform(-chaos_amplitude, chaos_amplitude, multiplicative_map.shape)
                            else: random_chaos = np.random.normal(0.0, chaos_amplitude / 3.0, multiplicative_map.shape)
                            multiplicative_map += random_chaos.astype(multiplicative_map.dtype)
                        multiplicative_map = np.clip(multiplicative_map, 0.0, None)

                        current_spec_ft = np.clip((spec_with_offset * multiplicative_map), 0.0, None)
                    # else: # Логируем несовпадение размеров (редко)
                    #     if random.random() < 0.01: warnings.warn(f"Perlin пропущен: несовпадение размеров карты/спектр. {file_id}")

                except IndexError: pass # Список карт пуст
                except Exception as e_perlin:
                    if random.random() < 0.01: warnings.warn(f"Ошибка Perlin для {file_id}: {e_perlin}. Пропуск.")

            # 3. Additive Gaussian Noise
            if self.apply_additive_gaussian and random.random() < self.add_gauss_apply_prob:
                try:
                    std_dev = random.uniform(self.add_gauss_config.get("std_dev_min", 0.05), self.add_gauss_config.get("std_dev_max", 0.15))
                    additive_noise = np.random.normal(0.0, std_dev, current_spec_ft.shape)
                    current_spec_ft = np.clip((current_spec_ft + additive_noise.astype(current_spec_ft.dtype)), 0.0, None)
                except Exception as e_add_gauss:
                    if random.random() < 0.01: warnings.warn(f"Ошибка Add Gauss для {file_id}: {e_add_gauss}. Пропуск.")

        # --- Конец блока if self.is_train ---

        # 4. Преобразование в тензор (C=1, T, F)
        features_ctf = np.expand_dims(current_spec_ft.T, axis=0)
        features_tensor = torch.tensor(features_ctf, dtype=torch.float32)

        # 5. Применение SpecAugment
        if self.apply_spec_augment and self.spec_augment_transform:
            try: features_tensor = self.spec_augment_transform(features_tensor)
            except Exception as e_specaug:
                if random.random() < 0.01: warnings.warn(f"Ошибка SpecAug для {file_id}: {e_specaug}. Пропуск.")

        # Проверка на NaN/Inf
        if torch.isnan(features_tensor).any() or torch.isinf(features_tensor).any():
            warnings.warn(f"NaN/Inf в финальном тензоре для {file_id}. Пропуск.")
            return None

        # 6. Кодирование текста
        morse_text_str = str(morse_text)
        blank_idx = self.ctc_config.get("blank_idx", 0)
        encoded_text_list = []
        for char in morse_text_str:
            idx = self.char_to_int.get(char)
            if idx is None:
                # if random.random() < 0.01: warnings.warn(f"Символ '{char}' не найден в словаре ({file_id}). Замена на бланк.")
                encoded_text_list.append(blank_idx)
            else:
                encoded_text_list.append(idx)
        encoded_text = torch.tensor(encoded_text_list, dtype=torch.long)

        # 7. Получение длин
        input_length = torch.tensor(features_tensor.shape[1], dtype=torch.long) # T
        target_length = torch.tensor(len(encoded_text), dtype=torch.long) # L

        if input_length.item() <= 0:
            warnings.warn(f"Нулевая длина входа (T=0) для {file_id}. Пропуск.")
            return None

        # (C, T, F), (L,), scalar, scalar
        return features_tensor, encoded_text, input_length, target_length


def collate_fn(batch: List[Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]],
               pad_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Обрабатывает батч данных из MorseDataset (v10.1 - Вынесен в модуль).
    """
    original_batch_size = len(batch)
    valid_batch = [b for b in batch if b is not None]
    filtered_count = original_batch_size - len(valid_batch)

    if not valid_batch:
        # if original_batch_size > 0: warnings.warn(f"Весь батч ({original_batch_size}) отфильтрован.")
        return None
    # if filtered_count > 0 and random.random() < 0.01: warnings.warn(f"Отфильтровано {filtered_count}/{original_batch_size} None-сэмплов.")

    try:
        features, texts, input_lengths, target_lengths = zip(*valid_batch)
    except ValueError as e: print(f"❌ Ошибка распаковки батча: {e}"); return None

    if not features: return None

    # Проверка консистентности C и F, и T > 0
    num_channels = features[0].shape[0]; num_freqs = features[0].shape[2]
    valid_indices = [i for i, f in enumerate(features) if f.dim() == 3 and f.shape[0] == num_channels and f.shape[2] == num_freqs and f.shape[1] > 0]
    invalid_shapes_found = len(features) - len(valid_indices)

    # if invalid_shapes_found > 0 and random.random() < 0.05: warnings.warn(f"Отфильтровано {invalid_shapes_found} сэмплов из-за некорректной формы.")

    if not valid_indices: return None

    # Отбор валидных
    valid_features = [features[i] for i in valid_indices]
    valid_texts = [texts[i] for i in valid_indices]
    valid_input_lengths = [input_lengths[i] for i in valid_indices]
    valid_target_lengths = [target_lengths[i] for i in valid_indices]

    # Паддинг признаков (C, T, F) -> (T, C, F) -> pad -> (T_max, B, C, F) -> (B, C, T_max, F)
    features_permuted = [f.permute(1, 0, 2) for f in valid_features]
    features_padded = pad_sequence(features_permuted, batch_first=False, padding_value=0.0)
    features_padded = features_padded.permute(1, 2, 0, 3)

    # Паддинг текстов (L,) -> pad -> (B, L_max)
    texts_padded = pad_sequence(valid_texts, batch_first=True, padding_value=pad_idx)

    # Стекинг длин -> (B,)
    input_lengths_tensor = torch.stack(valid_input_lengths)
    target_lengths_tensor = torch.stack(valid_target_lengths)

    # Финальная проверка
    final_batch_size = features_padded.shape[0]
    if not (final_batch_size == texts_padded.shape[0] == input_lengths_tensor.shape[0] == target_lengths_tensor.shape[0]):
         print(f"❌ Ошибка collate_fn: Несоответствие размеров батча!")
         return None

    return features_padded, texts_padded, input_lengths_tensor, target_lengths_tensor