# src/inference/predict.py

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast # Для инференса с AMP
import pandas as pd
import numpy as np # Может понадобиться
from pathlib import Path
from tqdm.notebook import tqdm # Или from tqdm import tqdm
import gc
import traceback
from collections import OrderedDict
from typing import Optional, Dict, List

# --- Импорты из других модулей src ---
from src.models.architectures import MorseRecognizer # Нужна модель
from src.data_processing.datasets import MorseDataset, collate_fn # Нужен датасет и коллэйт
from src.training.evaluate import ctc_greedy_decode # Нужен декодер

# --- Сигнатура функции ---
def generate_submission(
    config: Dict,
    model_path: str,
    test_df: pd.DataFrame, # DataFrame с тестовыми ID и путями к файлам
    char_to_int: Dict[str, int], # Для MorseDataset
    int_to_char: Dict[int, str], # Для декодирования
    submission_filename: str = "submission.csv", # Имя файла для сохранения
    # --- Дополнительные аргументы ---
    IS_MLFLOW_ACTIVE: bool = False, # Статус MLflow для логирования артефакта
    project_root: Optional[Path] = None # Путь к корню проекта
) -> bool:
    """
    Генерирует файл submission.csv для тестового набора данных.
    (Версия для src/inference/predict.py)

    Args:
        config (Dict): Словарь конфигурации.
        model_path (str): Путь к файлу обученной модели (.pth).
        test_df (pd.DataFrame): DataFrame с тестовыми данными (ID, full_path).
        char_to_int (Dict[str, int]): Словарь char -> int.
        int_to_char (Dict[int, str]): Словарь int -> char.
        submission_filename (str): Имя файла для сохранения submission.
        IS_MLFLOW_ACTIVE (bool): Флаг активности MLflow.
        project_root (Optional[Path]): Путь к корню проекта.

    Returns:
        bool: True, если submission успешно сгенерирован и сохранен, иначе False.
    """
    # ==========================================================================
    # Шаг 2: Инициализация и Настройка Инференса
    # ==========================================================================
    print(f"\n--- Инициализация для генерации Submission ---")
    print(f"  Используется модель: {model_path}")
    print(f"  Имя файла submission: {submission_filename}")

    model_file = Path(model_path)
    if not model_file.is_file(): # Проверяем, что это файл
        print(f"❌ ОШИБКА: Файл модели не найден или не является файлом: {model_path}")
        return False

    # --- Получение параметров из конфига ---
    try:
        device = torch.device(config.get("device", "cpu"))
        output_base_dir = project_root / config.get("paths", {}).get("output_dir", "output")
        run_description_safe = config.get('run_description', 'default_run').replace(":", "-").replace(" ", "_").replace("/", "_")
        output_dir = output_base_dir / run_description_safe # Директория конкретного запуска
        output_dir.mkdir(parents=True, exist_ok=True)

        text_column_name = config.get('morse_code_column', 'message') # Используется в MorseDataset
        test_file_column = config.get('test_file_column', 'id') # Для извлечения ID из test_df
        num_workers = config.get("num_workers", 0)
        ctc_blank_idx = config.get("ctc", {}).get("blank_idx", 0)
        ctc_pad_idx = config.get("ctc", {}).get("pad_idx", -1) # Нужен для collate_fn
        # Используем батчсайз из конфига (валидационный * 2 или трейновый * 2)
        inference_batch_size = config.get("finetuning", {}).get("batch_size", config.get("training",{}).get("batch_size", 16)) * 2
        inference_batch_size = max(1, inference_batch_size) # Не меньше 1
        # AMP для инференса (можно брать из training, т.к. обычно совпадает)
        use_amp = config.get("training", {}).get("use_amp", False)
    except Exception as e_conf:
        print(f"❌ ОШИБКА при получении параметров из конфига: {e_conf}")
        return False

    print(f"  Устройство: {device}")
    print(f"  Выходная директория запуска: {output_dir.resolve()}")
    print(f"  Batch Size для инференса: {inference_batch_size}")
    print(f"  Использовать AMP: {use_amp}")

    # --- Инициализация Модели ---
    print("\nИнициализация модели для инференса...")
    model = None # Инициализируем перед try
    try:
        model = MorseRecognizer(config).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        state_dict_to_load = None
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint: state_dict_to_load = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint: state_dict_to_load = checkpoint['state_dict']
        if state_dict_to_load is None: state_dict_to_load = checkpoint

        new_state_dict = OrderedDict()
        has_module_prefix = any(k.startswith('module.') for k in state_dict_to_load.keys())
        for k, v in state_dict_to_load.items():
            name = k[7:] if has_module_prefix and k.startswith('module.') else k
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
        model.eval() # Переводим модель в режим оценки!
        print("Модель успешно инициализирована и веса загружены.")
    except FileNotFoundError:
        print(f"❌ ОШИБКА: Файл модели не найден при загрузке: {model_path}")
        return False
    except Exception as e_load:
        print(f"❌ ОШИБКА загрузки/инициализации модели: {e_load}"); traceback.print_exc(limit=1); return False

    # --- Подготовка Тестового DataLoader ---
    print("\nПодготовка тестового DataLoader...")
    test_loader = None # Инициализируем перед try
    try:
        # Создаем тестовый Dataset БЕЗ аугментаций
        test_ds = MorseDataset(
            df=test_df, char_to_int=char_to_int, config=config, is_train=False,
            audio_augmenter=None, spec_augment_transform=None, # Аугментации выключены
            available_map_indices=None, project_root=project_root
        )
        collate_wrapper = lambda batch: collate_fn(batch, ctc_pad_idx)
        test_loader = DataLoader(
            test_ds, batch_size=inference_batch_size, shuffle=False, # НЕ перемешиваем тест
            num_workers=num_workers, collate_fn=collate_wrapper, pin_memory=True
        )
        print(f"Тестовый DataLoader создан: {len(test_loader)} батчей.")
        if len(test_loader) == 0 and len(test_df) > 0:
            print("!!! ПРЕДУПРЕЖДЕНИЕ: Тестовый DataLoader пуст, хотя test_df содержит данные!")
            # Не прерываем, возможно, все файлы в test_df некорректны
    except Exception as e_ds:
        print(f"❌ ОШИБКА при создании тестового DataLoader: {e_ds}"); traceback.print_exc(limit=1); return False

    # ==========================================================================
    # Шаг 3: Запуск Инференса
    # ==========================================================================
    print("\n--- Начало инференса на тестовом наборе ---")
    results = {} # Словарь для хранения {id: prediction}
    # Используем простой tqdm для консоли или notebook tqdm, если доступен
    try: from tqdm import tqdm
    except ImportError: from tqdm.notebook import tqdm

    pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Инференс", ncols=1000)
    processed_indices_count = 0 # Счетчик обработанных индексов

    with torch.no_grad(): # Отключаем расчет градиентов
        for batch_idx, batch_data in pbar:
            if batch_data is None:
                # print(f"Предупреждение (Инференс): Пропущен None батч {batch_idx}.") # Редко, можно убрать лог
                continue

            try:
                # Распаковываем только признаки, остальное игнорируем
                features, _, _, _ = batch_data
            except Exception as e_unpack:
                 print(f"❌ Ошибка распаковки батча {batch_idx} (Инференс): {e_unpack}"); continue

            if features is None or features.shape[0] == 0:
                # print(f"Предупреждение (Инференс): Пропущен батч {batch_idx} с пустыми признаками.") # Редко
                continue

            current_batch_size = features.shape[0]
            # Получаем ID файлов для этого батча, используя счетчик обработанных
            start_idx = processed_indices_count
            end_idx = start_idx + current_batch_size
            try:
                # Убедимся, что индексы не выходят за пределы test_df
                if end_idx > len(test_df):
                    print(f"Предупреждение (Инференс): Индекс {end_idx} выходит за пределы test_df ({len(test_df)}). Корректировка.")
                    end_idx = len(test_df)
                    current_batch_size = end_idx - start_idx # Обновляем размер батча
                    if current_batch_size <= 0: continue # Если не осталось валидных индексов

                batch_ids = test_df.iloc[start_idx:end_idx][test_file_column].tolist()
                processed_indices_count = end_idx # Обновляем счетчик
            except IndexError:
                 print(f"Ошибка (Инференс): Не удалось получить ID для батча {batch_idx} (индексы {start_idx}-{end_idx}). Пропуск."); continue
            except Exception as e_id:
                 print(f"Неизвестная ошибка при получении ID батча {batch_idx}: {e_id}"); continue


            features = features.to(device)
            batch_predictions = ["ERROR_UNKNOWN"] * current_batch_size # Предсказания по умолчанию

            try:
                # Прямой проход с AMP (если включено)
                with autocast(enabled=use_amp):
                    logits = model(features) # (B, T_out, V)

                # Проверка выхода модели
                if logits is None or logits.numel() == 0 or logits.shape[1] == 0:
                    print(f"Предупреждение (Инференс): Пустой выход модели для батча {batch_idx}.")
                    batch_predictions = [""] * current_batch_size # Предсказываем пустые строки
                else:
                    # Декодирование
                    decoded_preds = ctc_greedy_decode(logits.detach(), int_to_char, ctc_blank_idx)
                    if len(decoded_preds) == current_batch_size:
                        batch_predictions = decoded_preds
                    else:
                        print(f"Ошибка (Инференс): Несовпадение кол-ва предсказаний ({len(decoded_preds)}) и батча ({current_batch_size}) для батча {batch_idx}.")
                        batch_predictions = ["ERROR_DECODE_COUNT"] * current_batch_size

            except RuntimeError as e:
                 if "CUDA out of memory" in str(e): print(f"\n❌ OOM ОШИБКА (ИНФЕРЕНС, Батч: {batch_idx})!"); batch_predictions = ["ERROR_OOM"] * current_batch_size; gc.collect(); torch.cuda.empty_cache()
                 else: print(f"\n❌ RuntimeError (Инференс, Батч {batch_idx}): {e}"); batch_predictions = ["ERROR_RUNTIME"] * current_batch_size
            except Exception as e: print(f"\n❌ Ошибка (Инференс, Батч {batch_idx}): {e}"); batch_predictions = ["ERROR_INFERENCE"] * current_batch_size

            # Сохраняем результаты для батча
            for i, file_id in enumerate(batch_ids):
                # Проверяем, что индекс i не выходит за пределы batch_predictions
                if i < len(batch_predictions):
                    results[file_id] = batch_predictions[i]
                else:
                    print(f"Предупреждение: Индекс {i} вне диапазона предсказаний ({len(batch_predictions)}) для батча {batch_idx}.")
                    results[file_id] = "ERROR_PRED_INDEX"


            # Очистка памяти батча
            del features, logits, batch_data
            if 'decoded_preds' in locals(): del decoded_preds
            if batch_idx % 50 == 0: # Периодическая очистка GPU
                 gc.collect()
                 if device == torch.device('cuda'): torch.cuda.empty_cache()

    pbar.close()
    print("Инференс завершен.")

    # ==========================================================================
    # Шаг 4: Формирование и Сохранение Submission Файла
    # ==========================================================================
    submission_generated = False # Флаг успешной генерации
    try:
        # --- Проверка полноты результатов ---
        total_expected = len(test_df)
        total_obtained = len(results)
        print(f"\nПроверка результатов инференса:")
        print(f"  Ожидалось ID: {total_expected}")
        print(f"  Получено предсказаний: {total_obtained}")

        if total_obtained < total_expected:
            missing_ids = set(test_df[test_file_column]) - set(results.keys())
            num_missing = len(missing_ids)
            print(f"!!! ПРЕДУПРЕЖДЕНИЕ: {num_missing} ID отсутствуют в результатах инференса!")
            # Добавляем отсутствующие ID с ошибкой или пустым значением
            for mid in missing_ids:
                results.setdefault(mid, "") # Используем пустую строку для отсутствующих
            print(f"  Добавлены отсутствующие ID с пустыми предсказаниями.")
        elif total_obtained > total_expected:
             print(f"!!! ПРЕДУПРЕЖДЕНИЕ: Получено больше предсказаний ({total_obtained}), чем ожидалось ({total_expected}). Возможны дубликаты ID.")
             # Можно добавить логику удаления дубликатов, если необходимо

        print("\nФормирование файла submission...")
        # Создаем DataFrame из словаря результатов {id: prediction}
        results_df = pd.DataFrame(list(results.items()), columns=[test_file_column, text_column_name]) # Используем имя колонки из конфига

        # --- Обеспечение правильного порядка и полноты ---
        # Используем оригинальный test_df (только с колонкой ID) как основу
        # и присоединяем к нему результаты по ID файла.
        # Это гарантирует, что все ID из test_df будут в финальном файле
        # и в том же порядке, что и в sample_submission.
        final_submission_df = pd.merge(
            test_df[[test_file_column]], # Берем только колонку ID из оригинального test_df
            results_df,
            on=test_file_column,
            how='left' # Сохраняем все ID из левого DataFrame (test_df)
        )

        # Заполняем возможные пропуски (если merge не нашел ID в results_df, что маловероятно после проверки выше)
        # Используем пустую строку как значение по умолчанию для пропущенных предсказаний
        final_submission_df[text_column_name] = final_submission_df[text_column_name].fillna("")

        # Финальная проверка длины
        if len(final_submission_df) != len(test_df):
            print(f"!!! КРИТИЧЕСКОЕ ПРЕДУПРЕЖДЕНИЕ: Длина итогового submission ({len(final_submission_df)}) не совпадает с длиной test_df ({len(test_df)})!")
            # Можно добавить дополнительную диагностику здесь

        # --- Сохранение файла ---
        submission_path = output_dir / submission_filename
        final_submission_df[[test_file_column, text_column_name]].to_csv(submission_path, index=False) # Сохраняем только нужные колонки
        print(f"✅ Submission файл успешно сохранен: {submission_path.resolve()}")
        submission_generated = True # Устанавливаем флаг успеха

        # --- Логирование артефакта в MLflow (если активно) ---
        if IS_MLFLOW_ACTIVE and mlflow.active_run():
             try:
                 mlflow.log_artifact(str(submission_path), artifact_path="submission")
                 print(f"Артефакт submission '{submission_filename}' залогирован в MLflow.")
             except Exception as e_art_sub:
                 print(f"Предупреждение: Не удалось залогировать submission в MLflow: {e_art_sub}")

    except Exception as e_final:
        print(f"❌ ОШИБКА на этапе формирования/сохранения submission: {e_final}")
        traceback.print_exc(limit=1)
        submission_generated = False # Явно указываем на ошибку

    # --- Очистка памяти ---
    print("Очистка ресурсов после инференса...")
    vars_to_delete = ['model', 'test_loader', 'test_ds', 'results_df', 'final_submission_df', 'results']
    for var_name in vars_to_delete:
        if var_name in locals():
            del locals()[var_name]
    collected = gc.collect()
    print(f"  GC собрал {collected} объектов.")
    if device == torch.device('cuda') and torch.cuda.is_available():
        try: torch.cuda.empty_cache(); print("  Кэш CUDA очищен.")
        except Exception as e_cuda_clear: print(f"  Предупреждение: Не удалось очистить кэш CUDA: {e_cuda_clear}")
    print("Очистка завершена.")

    # --- Завершение функции ---
    print(f"\n>> Завершение generate_submission <<")
    return submission_generated # Возвращаем флаг успеха