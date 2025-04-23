import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import random
import warnings
from typing import Optional, Dict
import audiomentations # Нужно для type hinting

def get_features(
    audio_path: Path,
    audio_config: Dict,
    augmenter: Optional[audiomentations.Compose] = None,
    apply_audio_aug_prob: float = 0.0,
    is_train: bool = False
) -> Optional[np.ndarray]:
    """
    Загружает аудио, опционально применяет аудио аугментации,
    извлекает ЛИНЕЙНУЮ АМПЛИТУДНУЮ спектрограмму STFT (|STFT|).

    Args:
        audio_path: Путь к аудиофайлу.
        audio_config: Словарь с параметрами аудио (sample_rate, n_fft, hop_length).
        augmenter: Скомпилированный объект audiomentations.Compose или None.
        apply_audio_aug_prob: Вероятность применения augmenter (если он не None).
        is_train: Флаг для вывода логов.

    Returns:
        np.ndarray формы (F, T) с амплитудной спектрограммой или None в случае ошибки.
    """
    try:
        if not audio_path.is_file():
            # Не выводим предупреждение для каждого отсутствующего файла в val/test
            # if is_train: warnings.warn(f"Файл не найден {audio_path}. Пропуск.")
            return None

        y, current_sr = sf.read(audio_path, dtype='float32')
        target_sr = audio_config.get("sample_rate", 8000) # Безопасный доступ

        if y is None or y.shape[0] == 0:
            if is_train: warnings.warn(f"Пустой аудио сигнал в {audio_path.name}. Пропуск.")
            return None

        if current_sr != target_sr:
            y = librosa.resample(y=y, orig_sr=current_sr, target_sr=target_sr)
            if y is None or y.shape[0] == 0:
                if is_train: warnings.warn(f"Пустой сигнал после ресемплинга {audio_path.name}. Пропуск.")
                return None

        if is_train and augmenter and random.random() < apply_audio_aug_prob:
            try:
                y_aug = augmenter(samples=y, sample_rate=target_sr)
                if y_aug is not None and y_aug.shape[0] > 0:
                    y = y_aug
                elif is_train: # Логируем только если аугментация вернула пустоту
                    warnings.warn(f"Аугментация вернула пустой массив для {audio_path.name}. Используется оригинал.")
            except Exception as e_aug:
                 if is_train: warnings.warn(f"Ошибка аудио-аугментации для {audio_path.name}: {e_aug}. Используется оригинал.")

        stft_result = librosa.stft(
            y=y,
            n_fft=audio_config.get("n_fft", 512),
            hop_length=audio_config.get("hop_length", 96),
        )
        features_ft = np.abs(stft_result) # (F, T)

        if np.isnan(features_ft).any() or np.isinf(features_ft).any():
             if is_train: warnings.warn(f"NaN/Inf обнаружены в спектрограмме для {audio_path.name}. Пропуск.")
             return None

        if features_ft.shape[1] == 0:
            if is_train: warnings.warn(f"Нулевая длина признаков (T=0) для {audio_path.name}. Пропуск.")
            return None

        return features_ft.astype(np.float32)

    except sf.SoundFileError as e_sf:
         # Эти ошибки частые, если файл битый, логируем реже
         # if is_train and random.random() < 0.01: warnings.warn(f"SoundFile ошибка {audio_path.name}: {e_sf}. Пропуск.")
         return None
    except Exception as e:
        # Ловим все остальные ошибки
        print(f"❌ Ошибка обработки файла {audio_path.name} в get_features: {e}")
        # traceback.print_exc(limit=1) # Раскомментировать для детальной отладки
        return None

