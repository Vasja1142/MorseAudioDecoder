# src/training/pipeline.py

# --- Стандартные и сторонние библиотеки ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR

import audiomentations
import mlflow
import pandas as pd
import numpy as np
from pathlib import Path
import time
import gc
import traceback
import contextlib
import math
import re
from collections import OrderedDict
from typing import Optional, List, Tuple, Dict, Union

# --- Импорты из других модулей проекта ---
try:
    # Убедитесь, что эти импорты соответствуют вашей структуре src/
    from src.models.architectures import MorseRecognizer
    from src.data_processing.datasets import MorseDataset, collate_fn
    from src.data_processing.augmentations import SpecAugmentTransform
    from .engine import train_epoch, evaluate_epoch # Относительный импорт engine
except ImportError as e:
    print(f"КРИТИЧЕСКАЯ ОШИБКА ИМПОРТА в pipeline.py: {e}")
    print("Убедитесь, что структура проекта верна и все зависимости установлены.")
    raise


# --- Сигнатура основной функции ---
def run_training(
    mode: str,                           # 'train' или 'finetune'
    config: Dict,                        # Словарь конфигурации эксперимента
    train_df: pd.DataFrame,              # DataFrame с обучающими данными
    val_df: pd.DataFrame,                # DataFrame с валидационными данными
    char_to_int: Dict[str, int],         # Словарь: символ -> индекс
    int_to_char: Dict[int, str],         # Словарь: индекс -> символ
    audio_augmenter_global: Optional[audiomentations.Compose], # Объект аудио-аугментатора
    available_map_indices: Optional[List[int]], # Индексы доступных карт Перлина
    checkpoint_path: Optional[str] = None, # Путь к чекпоинту для finetune
    IS_MLFLOW_ACTIVE: bool = False,      # Флаг активности MLflow
    project_root: Optional[Path] = None  # Корневая директория проекта
) -> Tuple[str, float]:
    """
    Запускает цикл обучения ('train') или дообучения ('finetune') модели.
    Возвращает путь к лучшей модели и её метрику (Levenshtein).
    """
    # --- Начало тела функции (Отступ 1 уровень) ---
    best_model_path: str = ""
    best_val_levenshtein: float = float('inf')
    start_epoch: int = 0
    epochs_no_improve: int = 0
    model, optimizer, criterion, scaler, scheduler = None, None, None, None, None
    train_loader, val_loader = None, None
    spec_augmenter = None
    output_dir: Path = Path("./outputs/default_run") # Значение по умолчанию

    print(f"\n>> Запуск run_training (mode='{mode}') <<")

    # --- Проверка режима и чекпоинта ---
    is_finetune_stage = (mode == 'finetune')
    absolute_checkpoint_path: Optional[str] = None

    if project_root is None:
        project_root = Path.cwd().resolve() # Используем CWD, если корень не передан
        print(f"Предупреждение: project_root не передан, используется CWD: {project_root}")

    if is_finetune_stage:
        if not checkpoint_path:
            print(f"❌ ОШИБКА ({mode.upper()}): Режим 'finetune' требует 'checkpoint_path'.")
            return "", float('inf')
        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.is_absolute():
            checkpoint_file = (project_root / checkpoint_file).resolve()
        if not checkpoint_file.exists():
            print(f"❌ ОШИБКА ({mode.upper()}): Чекпоинт не найден: {checkpoint_file}")
            return "", float('inf')
        absolute_checkpoint_path = str(checkpoint_file)
        print(f"Режим дообучения. Базовая модель: {checkpoint_file.name}")
    elif mode != 'train':
        print(f"❌ ОШИБКА: Неизвестный режим '{mode}'. Допустимы 'train' или 'finetune'.")
        return "", float('inf')

    # --- Основной блок try...except...finally ---
    try:
        # --- Начало блока try (Отступ 2 уровня) ---

        # --- Шаг 2.1: Инициализация и Настройка из Config ---
        print(f"\n--- Инициализация параметров для этапа '{mode.upper()}' ---")
        run_config = config.get(mode, config.get("training", {}))
        if not isinstance(run_config, dict):
            print(f"❌ ОШИБКА: Не найдена или некорректна секция '{mode}' или 'training' в config.")
            return "", float('inf')

        # Извлечение параметров
        device = torch.device(config.get("device", "cpu"))
        text_column_name = config.get('morse_code_column', 'message')
        apply_audio_augment_stage = run_config.get("apply_audio_augmentation", False)
        current_epochs = run_config.get('epochs', 1)
        current_batch_size = run_config.get('batch_size', 16) # Должно быть числом!
        current_lr = run_config.get('learning_rate', 0.0003)
        optimizer_name = run_config.get("optimizer", "AdamW")
        scheduler_name_from_config = run_config.get("scheduler", "NoSched")
        current_weight_decay = run_config.get('weight_decay', 0.0001)
        current_use_amp = run_config.get('use_amp', True)
        grad_accum_steps = max(1, run_config.get('gradient_accumulation_steps', 1))
        grad_clip_norm = run_config.get('gradient_clip_val', 1.0)
        early_stopping_patience = run_config.get('early_stopping_patience', 10)
        num_workers = config.get("num_workers", 0)
        ctc_blank_idx = config.get("ctc", {}).get("blank_idx", 0)
        ctc_pad_idx = config.get("ctc", {}).get("pad_idx", -1)

        apply_perlin_online = config.get("perlin_augmentation", {}).get("apply", False) and available_map_indices
        apply_additive_gaussian_online = config.get("additive_gaussian_augmentation", {}).get("apply", False)
        apply_spec_augment_online = config.get("spec_augmentation", {}).get("apply", False)

        # Определение выходной директории
        output_base_dir = project_root / config.get("paths", {}).get("output_dir", "outputs")
        run_description_safe = config.get('run_description', 'default_run').replace(":", "-").replace(" ", "_").replace("/", "_")
        output_dir_run_str = config.get("execution_info", {}).get("output_dir_run")
        if output_dir_run_str and Path(output_dir_run_str).is_dir():
             output_dir = Path(output_dir_run_str)
        else:
             output_dir = output_base_dir / run_description_safe
             if not output_dir_run_str:
                 print(f"Предупреждение: Путь к директории запуска не найден в config['execution_info'], используется: {output_dir}")
             output_dir.mkdir(parents=True, exist_ok=True)

        # Вывод ключевых параметров этапа
        print(f"  Device: {device}")
        print(f"  Output Dir (для чекпоинтов этапа): {output_dir.resolve()}")
        # ... (можно добавить больше print'ов при необходимости)

        # --- Шаг 2.2: Логирование параметров в MLflow ---
        if IS_MLFLOW_ACTIVE:
            print(f"\nMLflow: Логирование для этапа '{mode}' активно.")
            try:
                params_stage = {f"{mode}_{k}": v for k, v in run_config.items() if isinstance(v, (str, int, float, bool))}
                mlflow.log_params(params_stage)
                params_common = {
                    "run_description": config.get("run_description", "N/A"),
                    "run_mode_config": config.get("run_mode", "N/A"),
                    "current_stage": mode,
                    "base_model_for_finetune": Path(absolute_checkpoint_path).name if is_finetune_stage and absolute_checkpoint_path else "N/A",
                    "device": str(device),
                    "num_available_perlin_maps": len(available_map_indices) if available_map_indices else 0,
                    "apply_perlin_online": apply_perlin_online,
                    "apply_additive_gaussian_online": apply_additive_gaussian_online,
                    "apply_spec_augment_online": apply_spec_augment_online,
                    "random_seed": config.get("random_seed", "N/A")
                }
                mlflow.log_params(params_common)
                mlflow.log_param("scheduler_configured", scheduler_name_from_config)
            except Exception as e_mlflow_param:
                 print(f"Предупреждение: Ошибка логирования параметров в MLflow: {e_mlflow_param}")
        else:
            print(f"\nMLflow: Логирование для '{mode}' неактивно.")

        # --- Шаг 2.3: Инициализация Модели ---
        print("\nИнициализация модели...")
        try:
            model = MorseRecognizer(config).to(device)
            print(f"Модель '{type(model).__name__}' инициализирована.")
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Всего параметров: {total_params:,}")
            print(f"  Обучаемых параметров: {trainable_params:,}")
        except Exception as e_model_init:
            print(f"❌ ОШИБКА инициализации модели: {e_model_init}")
            raise

        # --- Шаг 2.4: Загрузка Чекпоинта (для finetune) ---
        if is_finetune_stage and absolute_checkpoint_path:
            print(f"\nЗагрузка состояния модели из: {Path(absolute_checkpoint_path).name}")
            try:
                checkpoint = torch.load(absolute_checkpoint_path, map_location=device)
                state_dict_to_load = None
                if 'model_state_dict' in checkpoint:
                    state_dict_to_load = checkpoint['model_state_dict']
                    print("  Найден ключ 'model_state_dict'.")
                elif isinstance(checkpoint, dict) and all(isinstance(k, str) for k in checkpoint.keys()) and not any(k in ['epoch', 'optimizer_state_dict'] for k in checkpoint.keys()):
                     state_dict_to_load = checkpoint
                     print("  Чекпоинт похож на чистый state_dict.")
                else:
                    raise KeyError("Ключ 'model_state_dict' не найден в чекпоинте и структура не распознана.")
                model.load_state_dict(state_dict_to_load)
                print("  Веса модели успешно загружены.")
                # (Опционально) Загрузка эпохи
                if 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch'] # Начнем со следующей эпохи
                    print(f"  Обучение будет возобновлено с эпохи {start_epoch + 1}")
            except FileNotFoundError:
                 print(f"❌ КРИТИЧЕСКАЯ ОШИБКА: Файл чекпоинта не найден: {absolute_checkpoint_path}")
                 raise
            except Exception as e_load:
                print(f"❌ ОШИБКА загрузки чекпоинта: {e_load}")
                traceback.print_exc(limit=1)
                raise

        # --- Шаг 2.5: Инициализация Критерия, Оптимизатора, Scaler ---
        print("\nИнициализация Criterion, Optimizer, Scaler...")
        try:
            criterion = nn.CTCLoss(blank=ctc_blank_idx, reduction='mean', zero_infinity=True)
            print(f"  Criterion: CTCLoss (blank={ctc_blank_idx})")
            if optimizer_name.lower() == "adamw":
                optimizer = optim.AdamW(model.parameters(), lr=current_lr, weight_decay=current_weight_decay)
            elif optimizer_name.lower() == "adam":
                optimizer = optim.Adam(model.parameters(), lr=current_lr, weight_decay=current_weight_decay)
            else:
                print(f"Предупреждение: Неизвестный оптимизатор '{optimizer_name}'. Используется AdamW.")
                optimizer = optim.AdamW(model.parameters(), lr=current_lr, weight_decay=current_weight_decay)
            print(f"  Optimizer: {type(optimizer).__name__} (lr={current_lr:.2e}, wd={current_weight_decay:.2e})")
            scaler = GradScaler(enabled=(device.type == 'cuda' and current_use_amp))
            print(f"  GradScaler: {'Включен' if scaler.is_enabled() else 'Выключен'}")
            # (Опционально) Загрузка состояния оптимизатора после его инициализации
            if is_finetune_stage and 'optimizer_state_dict' in checkpoint and optimizer:
                 try:
                     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                     print("  Состояние оптимизатора загружено.")
                 except Exception as e_optim_load:
                     print(f"  Предупреждение: Не удалось загрузить состояние оптимизатора: {e_optim_load}")
        except Exception as e_optim_init:
            print(f"❌ ОШИБКА инициализации Criterion/Optimizer/Scaler: {e_optim_init}")
            raise

        # --- Шаг 2.6: Инициализация SpecAugment Transform ---
        spec_augmenter = None
        if apply_spec_augment_online:
            print("\nИнициализация SpecAugment...")
            try:
                spec_augment_config = config.get("spec_augmentation", {})
                if spec_augment_config.get("time_mask_param") or spec_augment_config.get("freq_mask_param"):
                    spec_augmenter = SpecAugmentTransform(spec_augment_config)
                    print("  SpecAugmentTransform создан.")
                else:
                    print("  Параметры SpecAugment не заданы, трансформация не создана.")
                    apply_spec_augment_online = False
            except Exception as e_spec:
                print(f"❌ ОШИБКА создания SpecAugmentTransform: {e_spec}")
                apply_spec_augment_online = False
        else:
            print("\nSpecAugment НЕ будет применяться.")

        # --- Шаг 2.7: Создание Датасетов и Даталоадеров ---
        print(f"\nПодготовка данных (Dataset/DataLoader) для '{mode}'...")
        train_ds, val_ds = None, None
        try:
            # --- ИСПРАВЛЕННЫЙ ВЫЗОВ ДЛЯ train_ds ---
            train_ds = MorseDataset(
                df=train_df,
                char_to_int=char_to_int,
                config=config,
                is_train=True,
                audio_augmenter=(audio_augmenter_global if apply_audio_augment_stage else None),
                spec_augment_transform=(spec_augmenter if apply_spec_augment_online else None),
                # apply_perlin_on_fly=apply_perlin_online, # <-- УДАЛЕНО
                # apply_additive_gaussian_on_fly=apply_additive_gaussian_online, # <-- УДАЛЕНО
                available_map_indices=available_map_indices,
                project_root=project_root
            )
            # --- ИСПРАВЛЕННЫЙ ВЫЗОВ ДЛЯ val_ds ---
            val_ds = MorseDataset(
                df=val_df,
                char_to_int=char_to_int,
                config=config,
                is_train=False,
                audio_augmenter=None,
                spec_augment_transform=None,
                # apply_perlin_on_fly=False, # <-- УДАЛЕНО
                # apply_additive_gaussian_on_fly=False, # <-- УДАЛЕНО
                available_map_indices=None,
                project_root=project_root
            )
            print(f"  Размеры Dataset: Train={len(train_ds)}, Val={len(val_ds)}")
            if len(train_ds) == 0 or len(val_ds) == 0:
                 raise ValueError("Один из датасетов (train или val) пуст!")
        except TypeError as e_ds_type:
             print(f"❌ ОШИБКА TypeError при создании MorseDataset: {e_ds_type}")
             # Добавим подсказку, если ошибка все еще та же
             if 'unexpected keyword argument' in str(e_ds_type):
                 print("   -> Проверьте определение __init__ в src/data_processing/datasets.py")
                 print("   -> Убедитесь, что все передаваемые аргументы ожидаются конструктором.")
             raise
        except Exception as e_ds:
             print(f"❌ ОШИБКА при создании MorseDataset: {e_ds}")
             traceback.print_exc(limit=1)
             raise

        # --- Дальнейший код создания DataLoader'ов остается без изменений ---
        try:
            collate_wrapper = lambda batch: collate_fn(batch, ctc_pad_idx)
            # Проверка current_batch_size перед использованием
            if not isinstance(current_batch_size, int) or current_batch_size <= 0:
                 raise ValueError(f"Некорректное значение current_batch_size: {current_batch_size}. Ожидалось положительное целое число.")
            safe_batch_size = max(1, int(current_batch_size))
            train_loader = DataLoader(
                train_ds, batch_size=safe_batch_size, shuffle=True, num_workers=num_workers,
                collate_fn=collate_wrapper, pin_memory=True, drop_last=True
            )
            val_batch_size = max(1, safe_batch_size * 2)
            val_loader = DataLoader(
                val_ds, batch_size=val_batch_size, shuffle=False, num_workers=num_workers,
                collate_fn=collate_wrapper, pin_memory=True
            )
            print(f"  Количество батчей: Train={len(train_loader)}, Val={len(val_loader)}")
            if len(train_loader) == 0 or len(val_loader) == 0:
                 raise ValueError("Один из DataLoader'ов (train или val) пуст после создания!")
        except ValueError as e_dl_val:
             print(f"❌ ОШИБКА при создании DataLoader: {e_dl_val}")
             traceback.print_exc(limit=1)
             raise
        except Exception as e_dl:
             print(f"❌ ОШИБКА при создании DataLoader: {e_dl}")
             traceback.print_exc(limit=1)
             raise

        # --- Шаг 2.8: Инициализация Планировщика (Scheduler) ---
        print("\nИнициализация Scheduler...")
        scheduler = None
        scheduler_name = scheduler_name_from_config.strip().lower() if isinstance(scheduler_name_from_config, str) else "nosched"
        scheduler_params = config.get("schedulers", {})

        try:
            if scheduler_name == "reducelronplateau":
                params = scheduler_params.get("ReduceLROnPlateau", {})
                factor = params.get(f"factor_{mode}", params.get("factor", 0.5))
                patience = params.get(f"patience_{mode}", params.get("patience", 5))
                min_lr = params.get("min_lr", 1e-7)
                verbose = params.get("verbose", True)
                scheduler = ReduceLROnPlateau(
                    optimizer, mode='min', factor=factor, patience=patience,
                    min_lr=min_lr, verbose=verbose
                )
                print(f"  Scheduler: ReduceLROnPlateau (factor={factor}, patience={patience})")
            elif scheduler_name == "onecyclelr":
                 params = scheduler_params.get("OneCycleLR", {})
                 if not train_loader: raise ValueError("DataLoader не создан перед OneCycleLR")
                 total_steps = len(train_loader) * current_epochs // grad_accum_steps
                 max_lr = current_lr
                 pct_start = params.get("pct_start", 0.1)
                 anneal_strategy = params.get("anneal_strategy", "cos")
                 div_factor = params.get("div_factor", 5.0)
                 final_div_factor = params.get("final_div_factor", 1000.0)
                 verbose = params.get("verbose", False)
                 scheduler = OneCycleLR(
                     optimizer, max_lr=max_lr, total_steps=total_steps,
                     pct_start=pct_start, anneal_strategy=anneal_strategy,
                     div_factor=div_factor, final_div_factor=final_div_factor,
                     verbose=verbose
                 )
                 print(f"  Scheduler: OneCycleLR (max_lr={max_lr:.2e}, total_steps={total_steps})")
            elif scheduler_name in ["nosched", "none", ""]:
                print("  Scheduler: Не используется.")
            else:
                print(f"Предупреждение: Неизвестный scheduler '{scheduler_name}'. Не используется.")

            if IS_MLFLOW_ACTIVE and mlflow.active_run():
                with contextlib.suppress(Exception):
                    mlflow.log_param(f"{mode}_scheduler_initialized", type(scheduler).__name__ if scheduler else "None")
        except Exception as e_sched_init:
            print(f"❌ ОШИБКА инициализации Scheduler: {e_sched_init}")
            scheduler = None

        # ==========================================================================
        # Шаг 3: Основной Цикл Обучения
        # ==========================================================================
        print(f"\n--- НАЧАЛО ЦИКЛА ОБУЧЕНИЯ ({mode.upper()}) ---")
        print(f"Всего эпох для запуска: {current_epochs}, начиная с эпохи {start_epoch + 1}")

        if current_epochs <= start_epoch:
            print("Предупреждение: Количество эпох для запуска <= начальной эпохе. Цикл не будет выполнен.")

        epoch = start_epoch - 1 # Инициализация на случай, если цикл не выполнится

        # --- Основной цикл по эпохам ---
        for epoch in range(start_epoch, current_epochs):
            # --- Начало тела цикла for (Отступ 3 уровня) ---
            print("-" * 60)
            print(f"Epoch {epoch+1}/{current_epochs} ({mode.upper()})")
            epoch_start_time = time.time()
            model.train() # Режим обучения

            train_loss, train_lev, avg_lr = train_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=(scheduler if scheduler_name == "onecyclelr" else None), # Передаем только OneCycleLR
                scaler=scaler,
                device=device,
                config=config, # engine.py должен сам взять grad_accum и clip из config
                epoch_num=epoch,
                total_epochs=current_epochs,
                int_to_char=int_to_char,
                mode=mode
            )

            # --- 3.2 Эпоха Валидации ---
            model.eval() # Режим оценки
            val_loss, val_levenshtein, decoded_pairs = evaluate_epoch(
                model=model, dataloader=val_loader, criterion=criterion,
                device=device, config=config, int_to_char=int_to_char
            )
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time

            # --- 3.3 Вывод результатов эпохи ---
            current_lr_display = optimizer.param_groups[0]['lr']
            print(f"\nEpoch {epoch+1}/{current_epochs} Results | Duration: {epoch_duration:.2f} sec")
            print(f"  Train Loss: {train_loss:.4f} | Train Levenshtein: {train_lev:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val Levenshtein:   {val_levenshtein:.4f} (Best: {best_val_levenshtein:.4f})")
            print(f"  Current LR: {current_lr_display:.2e}")
            print("  Примеры декодирования (Предсказание | Реальность):")
            num_pairs_to_show = min(5, len(decoded_pairs))
            if num_pairs_to_show > 0:
                 for i in range(num_pairs_to_show):
                     pred_text, target_text = decoded_pairs[i]
                     print(f"    '{str(pred_text)[:80]}' | '{str(target_text)[:80]}'")
            else:
                 print("    (Нет декодированных пар для отображения)")

            # --- 3.4 Логирование в MLflow ---
            if IS_MLFLOW_ACTIVE and mlflow.active_run():
                metrics_to_log = {
                    f"{mode}_train_loss": train_loss, f"{mode}_train_lev": train_lev,
                    f"{mode}_val_loss": val_loss, f"{mode}_val_levenshtein": val_levenshtein,
                    f"{mode}_learning_rate": current_lr_display, f"{mode}_epoch_duration": epoch_duration,
                }
                valid_metrics = {k: float(v) for k, v in metrics_to_log.items() if isinstance(v, (int, float)) and np.isfinite(v)}
                if valid_metrics:
                    with contextlib.suppress(Exception): # Игнорируем ошибки логирования
                        mlflow.log_metrics(valid_metrics, step=epoch + 1)

            # --- 3.5 Шаг шедулера (если не OneCycleLR) ---
            if scheduler and isinstance(scheduler, ReduceLROnPlateau):
                metric_to_step = val_levenshtein if np.isfinite(val_levenshtein) else float('inf')
                scheduler.step(metric_to_step)

            # --- 3.6 Сохранение лучшей модели и Early Stopping ---
            is_best_epoch = False
            if np.isfinite(val_levenshtein) and val_levenshtein < best_val_levenshtein:
                print(f"  ✨ Val Levenshtein УЛУЧШИЛСЯ: {best_val_levenshtein:.4f} -> {val_levenshtein:.4f}")
                best_val_levenshtein = val_levenshtein
                epochs_no_improve = 0
                is_best_epoch = True
                model_save_name = f"{run_description_safe}_{mode.upper()}_best_epoch{epoch+1}_lev{val_levenshtein:.4f}.pth"
                model_save_path = output_dir / model_save_name
                try:
                    torch.save({
                                'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'val_levenshtein': best_val_levenshtein, 'config_snapshot': config
                               }, model_save_path)
                    best_model_path = str(model_save_path.resolve())
                    print(f"  Модель СОХРАНЕНА как: {model_save_path.name}")
                except Exception as e_save:
                    print(f"  ❌ ОШИБКА при сохранении модели: {e_save}")
                    best_model_path = ""
            else:
                epochs_no_improve += 1
                improvement_status = "не улучшился" if np.isfinite(val_levenshtein) else "был некорректен"
                print(f"  Val Levenshtein {improvement_status}. Счетчик Early Stop: {epochs_no_improve}/{early_stopping_patience}")

            if epochs_no_improve >= early_stopping_patience:
                print(f"\n❗️ Ранняя остановка ({mode.upper()}) сработала после {epoch + 1} эпох.")
                break # Выходим из цикла for epoch

            # --- 3.7 Очистка памяти в конце эпохи ---
            gc.collect()
            if device.type == 'cuda':
                 with contextlib.suppress(Exception):
                     torch.cuda.empty_cache()
        # --- КОНЕЦ ЦИКЛА for epoch --- (Отступ 2 уровня)

        # --- Шаг 3.8: Вывод и логирование финальных результатов этапа ---
        epochs_actually_run = epoch + 1
        if epochs_actually_run > start_epoch:
            print(f"\n--- Обучение ({mode.upper()}) ЗАВЕРШЕНО после {epochs_actually_run} эпох ---")
            if best_model_path and Path(best_model_path).exists():
                print(f"✅ Лучшая модель ({mode}) сохранена в: {Path(best_model_path).name}")
                print(f"   Лучший Val Levenshtein ({mode}): {best_val_levenshtein:.4f}")
            else:
                print(f"⚠️ Лучшая модель ({mode}) не была сохранена.")
            if IS_MLFLOW_ACTIVE and mlflow.active_run():
                try:
                    mlflow.log_metric(f"final_best_{mode}_val_levenshtein", best_val_levenshtein if np.isfinite(best_val_levenshtein) else -1.0)
                    mlflow.log_param(f"final_{mode}_epochs_run", epochs_actually_run)
                    mlflow.log_param(f"final_best_{mode}_model_path", Path(best_model_path).name if best_model_path else "N/A")
                    if best_model_path and Path(best_model_path).exists():
                         mlflow.log_artifact(best_model_path, artifact_path=f"best_model_{mode}")
                         print("Артефакт лучшей модели этапа залогирован в MLflow.")
                except Exception as e_art_final:
                    print(f"Предупреждение: Ошибка логирования финальных артефактов/метрик в MLflow: {e_art_final}")
        else:
            print(f"\nЦикл обучения ({mode.upper()}) не был запущен.")
            if IS_MLFLOW_ACTIVE and mlflow.active_run():
                with contextlib.suppress(Exception): mlflow.log_param(f"final_{mode}_epochs_run", 0)
    # --- КОНЕЦ БЛОКА try --- (Отступ 1 уровень)

    # ==========================================================================
    # Шаг 4: Обработка Ошибок и Очистка Ресурсов
    # ==========================================================================
    except KeyboardInterrupt:
        print(f"\n❗️ Обучение ({mode.upper()}) прервано пользователем (KeyboardInterrupt).")
    except ValueError as e_val: # Ловим ValueError отдельно (например, от DataLoader)
        print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА (ValueError) во время '{mode}': {e_val}")
        traceback.print_exc(limit=2)
        best_model_path = ""
        best_val_levenshtein = float('inf')
        if IS_MLFLOW_ACTIVE and mlflow.active_run():
             with contextlib.suppress(Exception): mlflow.set_tag("error_stage", mode)
    except Exception as e:
        print(f"\n❌ НЕПРЕДВИДЕННАЯ КРИТИЧЕСКАЯ ОШИБКА ({type(e).__name__}) во время '{mode}':")
        traceback.print_exc()
        best_model_path = ""
        best_val_levenshtein = float('inf')
        if IS_MLFLOW_ACTIVE and mlflow.active_run():
             with contextlib.suppress(Exception): mlflow.set_tag("error_stage", mode)
    finally:
        # --- Начало блока finally (Отступ 1 уровень) ---
        print(f"\nОчистка ресурсов после '{mode}'...")
        collected_items = gc.collect()
        print(f"  GC собрал {collected_items} объектов.")
        if 'device' in locals() and isinstance(device, torch.device) and device.type == 'cuda':
             if torch.cuda.is_available():
                 with contextlib.suppress(Exception):
                     torch.cuda.empty_cache()
                     print("  Кэш CUDA очищен.")
             else:
                 print("  CUDA недоступна, очистка кэша не требуется.")
        # Удаляем ссылки на большие объекты
        del model, optimizer, criterion, scaler, scheduler
        del train_loader, val_loader, train_ds, val_ds, spec_augmenter
        print("Очистка ресурсов завершена.")
        # --- Конец блока finally ---

    # --- Возвращаем результат (Отступ 1 уровень) ---
    print(f"\n>> Завершение run_training (mode='{mode}').")
    final_path_str = Path(best_model_path).name if best_model_path else "N/A"
    if np.isfinite(best_val_levenshtein):
        final_metric_str = f"{best_val_levenshtein:.4f}"
        final_metric = best_val_levenshtein
    else:
        final_metric_str = "inf"
        final_metric = float('inf')
    print(f"   Возвращаемый путь (лучшая модель этапа): '{final_path_str}'")
    print(f"   Возвращаемая метрика (лучший Val Levenshtein): {final_metric_str}")
    final_path_return = best_model_path if best_model_path and Path(best_model_path).exists() else ""
    return final_path_return, final_metric

# --- Конец файла src/training/pipeline.py ---