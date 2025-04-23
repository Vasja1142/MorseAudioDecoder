# src/training/pipeline.py

# --- Стандартные и сторонние библиотеки ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import OneCycleLR

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
from typing import Optional, List, Tuple, Dict, Any

# --- Импорты из других модулей проекта ---
try:
    from src.models.architectures import MorseRecognizer
    from src.data_processing.datasets import MorseDataset, collate_fn
    from src.data_processing.augmentations import SpecAugmentTransform
    from .engine import train_epoch, evaluate_epoch
    # Вспомогательная функция для логирования (можно вынести в mlflow_utils.py)
    def _log_mlflow_params(params: Dict[str, Any], prefix: Optional[str] = None):
        """Логирует параметры в MLflow, фильтруя None и добавляя префикс."""
        if not mlflow.active_run(): return
        filtered_params = {}
        for k, v in params.items():
            # Пропускаем сложные типы и None
            if isinstance(v, (str, int, float, bool)):
                key = f"{prefix}_{k}" if prefix else k
                filtered_params[key] = v
        if filtered_params:
            with contextlib.suppress(Exception):
                mlflow.log_params(filtered_params)

except ImportError as e:
    print(f"КРИТИЧЕСКАЯ ОШИБКА ИМПОРТА в pipeline.py: {e}"); raise

# --- Сигнатура основной функции ---
def run_training(
    mode: str, config: Dict[str, Any], train_df: pd.DataFrame, val_df: pd.DataFrame,
    char_to_int: Dict[str, int], int_to_char: Dict[int, str],
    audio_augmenter_global: Optional[audiomentations.Compose],
    available_map_indices: Optional[List[int]],
    IS_MLFLOW_ACTIVE: bool = False, project_root: Optional[Path] = None,
    output_dir_run: Optional[Path] = None
) -> Tuple[str, float]:
    """
    Запускает цикл обучения ('train') или дообучения ('finetune') модели.
    Возвращает путь к лучшей модели этапа и её Levenshtein score.
    """
    # --- 1. Инициализация и Проверки ---
    best_model_path_stage: str = ""
    best_val_levenshtein: float = float('inf')
    start_epoch: int = 0
    model, optimizer, criterion, scaler, scheduler = None, None, None, None, None
    train_loader, val_loader = None, None
    absolute_checkpoint_path: Optional[str] = None

    print(f"\n>> Запуск run_training (mode='{mode}') <<")

    if project_root is None: project_root = Path.cwd().resolve()
    if output_dir_run is None: print("❌ ОШИБКА: Не передан 'output_dir_run'."); return "", float('inf')
    output_dir_run.mkdir(parents=True, exist_ok=True)

    is_finetune_stage = (mode == 'finetune')
    if is_finetune_stage:
        checkpoint_path_rel = config.get("finetuning", {}).get("finetune_only_checkpoint_path")
        if not checkpoint_path_rel: print(f"❌ ОШИБКА ({mode}): Не указан 'finetuning.finetune_only_checkpoint_path'."); return "", float('inf')
        checkpoint_file = Path(checkpoint_path_rel)
        if not checkpoint_file.is_absolute(): checkpoint_file = (project_root / checkpoint_file).resolve()
        if not checkpoint_file.exists(): print(f"❌ ОШИБКА ({mode}): Чекпоинт не найден: {checkpoint_file}"); return "", float('inf')
        absolute_checkpoint_path = str(checkpoint_file)
        print(f"Режим дообучения. Базовая модель: {checkpoint_file.name}")
    elif mode != 'train': print(f"❌ ОШИБКА: Неизвестный режим '{mode}'."); return "", float('inf')

    # --- Основной блок try...except...finally ---
    try:
        # --- 2. Извлечение Параметров Конфигурации ---
        print(f"\n--- Чтение параметров для этапа '{mode.upper()}' ---")
        if mode == 'train':
            stage_config_key = 'training'
        elif mode == 'finetune':
            stage_config_key = 'finetuning' # Правильное имя секции
        else:
            # Эта проверка дублирует проверку в начале функции, но для надежности
            raise ValueError(f"Неизвестный режим '{mode}' при попытке получить ключ конфигурации.")

        run_config = config.get(stage_config_key)
        if not isinstance(run_config, dict): raise ValueError(f"Секция '{stage_config_key}' не найдена/некорректна.")

        # Параметры этапа
        device = torch.device(config.get("device", "cpu"))
        current_epochs = run_config.get('epochs', 1)
        current_batch_size = run_config.get('batch_size', 16)
        current_lr = run_config.get('learning_rate', 0.0003)
        optimizer_name = run_config.get("optimizer", "AdamW")
        scheduler_name = run_config.get("scheduler", "OneCycleLR") # Ожидаем только OneCycleLR или None/""
        current_weight_decay = run_config.get('weight_decay', 0.0001)
        current_use_amp = run_config.get('use_amp', True)
        grad_accum_steps = max(1, run_config.get('gradient_accumulation_steps', 1))
        grad_clip_norm = run_config.get('gradient_clip_val', 1.0)
        early_stopping_patience = run_config.get('early_stopping_patience', 10)
        apply_audio_augment_stage = run_config.get("apply_audio_augmentation", False)

        # Общие параметры
        num_workers = config.get("num_workers", 0)
        ctc_cfg = config.get("ctc", {})
        ctc_blank_idx = ctc_cfg.get("blank_idx", 0)
        ctc_pad_idx = ctc_cfg.get("pad_idx", -1)

        # Параметры онлайн-аугментаций
        apply_perlin = config.get("perlin_augmentation", {}).get("apply", False) and available_map_indices
        apply_add_gauss = config.get("additive_gaussian_augmentation", {}).get("apply", False)
        apply_spec_aug = config.get("spec_augmentation", {}).get("apply", False)

        print(f"  Device: {device}, Epochs: {current_epochs}, BS: {current_batch_size}, LR: {current_lr:.1e}")
        print(f"  Optimizer: {optimizer_name}, Scheduler: {scheduler_name}, AMP: {current_use_amp}")
        print(f"  Augment (Stage): Audio={apply_audio_augment_stage}")
        print(f"  Augment (Online): Perlin={apply_perlin}, AddGauss={apply_add_gauss}, SpecAug={apply_spec_aug}")

        # --- 3. Логирование в MLflow (если активно) ---
        if IS_MLFLOW_ACTIVE and mlflow.active_run():
            print(f"\nMLflow: Логирование параметров для этапа '{mode}'.")
            _log_mlflow_params(run_config, prefix=mode) # Параметры этапа
            _log_mlflow_params({ # Общие параметры
                "run_description": config.get("run_description"), "config_run_mode": config.get("run_mode"),
                "current_stage": mode, "base_model_finetune": Path(absolute_checkpoint_path).name if absolute_checkpoint_path else "N/A",
                "device": str(device), "num_perlin_maps": len(available_map_indices or []),
                "apply_perlin": apply_perlin, "apply_add_gauss": apply_add_gauss, "apply_spec_aug": apply_spec_aug,
                "random_seed": config.get("random_seed")
            })
            mlflow.log_param(f"{mode}_scheduler_configured", scheduler_name) # Какой шедулер настроен

        # --- 4. Инициализация Модели ---
        print("\nИнициализация модели...")
        model = MorseRecognizer(config).to(device)
        print(f"  Модель '{type(model).__name__}' создана.")
        # Опционально: вывод числа параметров
        # total_params = sum(p.numel() for p in model.parameters())
        # print(f"  Параметров: {total_params:,}")

        # --- 5. Загрузка Чекпоинта (для finetune) ---
        if is_finetune_stage and absolute_checkpoint_path:
            print(f"\nЗагрузка состояния модели из: {Path(absolute_checkpoint_path).name}")
            checkpoint = torch.load(absolute_checkpoint_path, map_location=device)
            state_dict_key = 'model_state_dict'
            if state_dict_key in checkpoint: state_dict_to_load = checkpoint[state_dict_key]
            elif isinstance(checkpoint, dict) and all(isinstance(k, str) for k in checkpoint.keys()): state_dict_to_load = checkpoint
            else: raise KeyError("Не удалось найти state_dict модели в чекпоинте.")
            model.load_state_dict(state_dict_to_load)
            print("  Веса модели успешно загружены.")
            # Информация из чекпоинта (не влияет на старт)
            loaded_epoch = checkpoint.get('epoch', 'N/A')
            loaded_lev = checkpoint.get('val_levenshtein', float('inf'))
            print(f"  (Инфо: чекпоинт эпохи {loaded_epoch}, lev={loaded_lev:.4f})")

        # --- 6. Инициализация Criterion, Optimizer, Scaler ---
        print("\nИнициализация Criterion, Optimizer, Scaler...")
        criterion = nn.CTCLoss(blank=ctc_blank_idx, reduction='mean', zero_infinity=True)
        if optimizer_name.lower() == "adamw": optimizer = optim.AdamW(model.parameters(), lr=current_lr, weight_decay=current_weight_decay)
        elif optimizer_name.lower() == "adam": optimizer = optim.Adam(model.parameters(), lr=current_lr, weight_decay=current_weight_decay)
        else: print(f"Предупреждение: Неизвестный оптимизатор '{optimizer_name}'. Используется AdamW."); optimizer = optim.AdamW(model.parameters(), lr=current_lr, weight_decay=current_weight_decay)
        scaler = GradScaler(enabled=(device.type == 'cuda' and current_use_amp))
        print(f"  Criterion, Optimizer ({type(optimizer).__name__}), Scaler ({'ON' if scaler.is_enabled() else 'OFF'}) инициализированы.")
        if is_finetune_stage: print("  (Состояние оптимизатора НЕ загружается для finetune)")

        # --- 6.1 Инициализация SpecAugment ---
        spec_augmenter = None
        if apply_spec_aug:
            print("\nИнициализация SpecAugment...")
            spec_cfg = config.get("spec_augmentation", {})
            if spec_cfg.get("time_mask_param", 0) > 0 or spec_cfg.get("freq_mask_param", 0) > 0:
                try: spec_augmenter = SpecAugmentTransform(spec_cfg); print("  SpecAugmentTransform создан.")
                except Exception as e_spec: print(f"❌ ОШИБКА создания SpecAugment: {e_spec}"); apply_spec_aug = False
            else: print("  Параметры SpecAugment не заданы."); apply_spec_aug = False
        else: print("\nSpecAugment НЕ будет применяться.")

        # --- 7. Создание Датасетов и Даталоадеров ---
        print(f"\nПодготовка данных (Dataset/DataLoader) для '{mode}'...")
        apply_all_augmentations_stage = run_config.get("apply_audio_augmentation", False)
        print(f"  Мастер-флаг аугментаций для этапа '{mode}': {apply_all_augmentations_stage}")

        # Создаем датасеты
        train_ds = MorseDataset(
            df=train_df, char_to_int=char_to_int, config=config, is_train=True,
            audio_augmenter=(audio_augmenter_global if apply_all_augmentations_stage else None),
            spec_augment_transform=(spec_augmenter if apply_all_augmentations_stage else None),
            available_map_indices=available_map_indices,
            project_root=project_root,
            apply_all_augmentations_flag=apply_all_augmentations_stage
        )
        val_ds = MorseDataset(
            df=val_df, char_to_int=char_to_int, config=config, is_train=False,
            audio_augmenter=None, spec_augment_transform=None,
            available_map_indices=None, project_root=project_root,
            apply_all_augmentations_flag=False
        )
        # Проверка созданных датасетов
        if not train_ds or not val_ds or len(train_ds) == 0 or len(val_ds) == 0:
            raise ValueError("Ошибка создания или пустой Dataset (train или val).")
        print(f"  Размеры Dataset: Train={len(train_ds)}, Val={len(val_ds)}")

        # Создаем DataLoader'ы
        collate_wrapper = lambda batch: collate_fn(batch, ctc_pad_idx)
        if not isinstance(current_batch_size, int) or current_batch_size <= 0:
            raise ValueError(f"Некорректное значение batch_size: {current_batch_size}")

        train_loader = DataLoader(
            train_ds, batch_size=current_batch_size, shuffle=True, num_workers=num_workers,
            collate_fn=collate_wrapper, pin_memory=(device.type == 'cuda'), drop_last=True,
            # Добавим persistent_workers, если num_workers > 0 и ОС позволяет
            persistent_workers=(num_workers > 0)
        )
        val_batch_size = max(1, current_batch_size * 2)
        val_loader = DataLoader(
            val_ds, batch_size=val_batch_size, shuffle=False, num_workers=num_workers,
            collate_fn=collate_wrapper, pin_memory=(device.type == 'cuda'),
            persistent_workers=(num_workers > 0)
        )

        if train_loader is None:
            raise ValueError("Ошибка: train_loader не был создан (равен None).")
        if val_loader is None:
            raise ValueError("Ошибка: val_loader не был создан (равен None).")
        # Проверяем длину только после проверки на None
        if len(train_loader) == 0:
             raise ValueError("Ошибка: train_loader пуст (не содержит батчей).")
        if len(val_loader) == 0:
             raise ValueError("Ошибка: val_loader пуст (не содержит батчей).")

        # --- 8. Инициализация Планировщика (Scheduler) ---
        print("\nИнициализация Scheduler...")
        scheduler = None
        scheduler_name_norm = str(scheduler_name).strip().lower() if scheduler_name else "none"

        if scheduler_name_norm == "onecyclelr":
             scheduler_params = config.get("schedulers", {}).get("OneCycleLR", {})
             steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
             total_steps = steps_per_epoch * current_epochs
             if total_steps <= 0: raise ValueError(f"Некорректный total_steps ({total_steps}) для OneCycleLR.")
             scheduler = OneCycleLR(optimizer, max_lr=current_lr, total_steps=total_steps, **scheduler_params)
             print(f"  Scheduler: OneCycleLR (max_lr={current_lr:.1e}, total_steps={total_steps})")
        elif scheduler_name_norm in ["nosched", "none", ""]: print("  Scheduler: Не используется.")
        else: print(f"Предупреждение: Указанный scheduler '{scheduler_name}' не поддерживается. Не используется.")

        if IS_MLFLOW_ACTIVE and mlflow.active_run():
            mlflow.log_param(f"{mode}_scheduler_initialized", type(scheduler).__name__ if scheduler else "None")

        # --- Конец части 1: Инициализация ---
        print("\n--- Инициализация завершена, переход к циклу обучения ---")


        # ==========================================================================
        # Шаг 9: Основной Цикл Обучения
        # ==========================================================================
        print(f"\n--- НАЧАЛО ЦИКЛА ОБУЧЕНИЯ ({mode.upper()}) ---")
        print(f"Всего эпох для запуска: {current_epochs}, начиная с эпохи {start_epoch + 1}")

        if current_epochs <= start_epoch:
            print("Предупреждение: Количество эпох <= начальной эпохе. Цикл не будет выполнен.")
            # Если чекпоинт был загружен, возвращаем его метрику
            if is_finetune_stage and 'val_levenshtein' in checkpoint:
                 best_val_levenshtein = checkpoint.get('val_levenshtein', float('inf'))
                 best_model_path_stage = absolute_checkpoint_path # Возвращаем исходный чекпоинт
                 print(f"Возвращается исходный чекпоинт с Levenshtein: {best_val_levenshtein:.4f}")
            # Иначе возвращаем ошибку/пустоту
            else:
                 return "", float('inf')

        # --- Основной цикл по эпохам ---
        for epoch in range(start_epoch, current_epochs):
            print("-" * 60)
            print(f"Epoch {epoch+1}/{current_epochs} ({mode.upper()})")
            epoch_start_time = time.time()

            # --- 9.1 Эпоха Обучения ---
            train_loss, avg_lr = train_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler, # <-- ИСПРАВЛЕНО: Передаем объект напрямую
                scaler=scaler,
                device=device,
                config=config,
                epoch_num=epoch,
                total_epochs=current_epochs,
                mode=mode
            )

            # --- 9.2 Эпоха Валидации ---
            # evaluate_epoch возвращает (avg_loss, avg_lev, decoded_pairs)
            val_loss, val_levenshtein, decoded_pairs = evaluate_epoch(
                model=model, dataloader=val_loader, criterion=criterion,
                device=device, config=config, int_to_char=int_to_char
            )
            epoch_duration = time.time() - epoch_start_time

            # --- 9.3 Вывод результатов эпохи ---
            print(f"\nEpoch {epoch+1}/{current_epochs} Results | Duration: {epoch_duration:.2f} sec")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val Levenshtein:   {val_levenshtein:.4f} (Best: {best_val_levenshtein:.4f})")
            print(f"  Current LR: {avg_lr:.2e}") # Используем средний LR за эпоху
            # Вывод примеров декодирования
            num_pairs_to_show = min(3, len(decoded_pairs)) # Меньше примеров для краткости
            if num_pairs_to_show > 0:
                 print("  Примеры декодирования (Предсказание | Реальность):")
                 for i in range(num_pairs_to_show):
                     pred_text, target_text = decoded_pairs[i]
                     print(f"    '{str(pred_text)[:60]}' | '{str(target_text)[:60]}'") # Короче строки

            # --- 9.4 Логирование в MLflow ---
            if IS_MLFLOW_ACTIVE and mlflow.active_run():
                metrics_to_log = {
                    f"{mode}_train_loss": train_loss,
                    f"{mode}_val_loss": val_loss,
                    f"{mode}_val_levenshtein": val_levenshtein,
                    f"{mode}_learning_rate": avg_lr, # Логируем средний LR
                    f"{mode}_epoch_duration": epoch_duration,
                }
                # Фильтруем NaN/Inf перед логированием
                valid_metrics = {k: float(v) for k, v in metrics_to_log.items() if isinstance(v, (int, float)) and np.isfinite(v)}
                if valid_metrics:
                    with contextlib.suppress(Exception):
                        mlflow.log_metrics(valid_metrics, step=epoch + 1)

            # --- 9.5 Шаг шедулера (ReduceLROnPlateau удален) ---
            # OneCycleLR шагает внутри train_epoch

            # --- 9.6 Сохранение лучшей модели и Early Stopping ---
            if np.isfinite(val_levenshtein) and val_levenshtein < best_val_levenshtein:
                print(f"  ✨ Val Levenshtein УЛУЧШИЛСЯ: {best_val_levenshtein:.4f} -> {val_levenshtein:.4f}")
                best_val_levenshtein = val_levenshtein
                epochs_no_improve = 0
                # Формируем имя файла
                model_save_name = f"{config.get('mlflow',{}).get('run_name_prefix', 'model')}_{mode.upper()}_epoch{epoch+1}_lev{val_levenshtein:.4f}.pth"
                model_save_path = output_dir_run / model_save_name
                try:
                    # Сохраняем только state_dict модели для экономии места
                    torch.save(model.state_dict(), model_save_path)
                    # Обновляем путь к лучшей модели ЭТОГО этапа
                    best_model_path_stage = str(model_save_path.resolve())
                    print(f"  Модель СОХРАНЕНА как: {model_save_path.name}")
                except Exception as e_save:
                    print(f"  ❌ ОШИБКА при сохранении модели: {e_save}")
                    best_model_path_stage = "" # Сбрасываем путь, если сохранение не удалось
            else:
                epochs_no_improve += 1
                improvement_status = "не улучшился" if np.isfinite(val_levenshtein) else "был некорректен"
                print(f"  Val Levenshtein {improvement_status}. Счетчик Early Stop: {epochs_no_improve}/{early_stopping_patience}")

            # Проверка Early Stopping
            if epochs_no_improve >= early_stopping_patience:
                print(f"\n❗️ Ранняя остановка ({mode.upper()}) сработала после {epoch + 1} эпох.")
                break # Выходим из цикла for epoch

            # Очистка памяти в конце эпохи
            gc.collect()
            if device.type == 'cuda': torch.cuda.empty_cache()
        # --- КОНЕЦ ЦИКЛА for epoch ---

        # --- Шаг 10: Вывод и логирование финальных результатов этапа ---
        epochs_actually_run = epoch + 1 # Учитываем последнюю выполненную эпоху
        if epochs_actually_run > start_epoch:
            print(f"\n--- Обучение ({mode.upper()}) ЗАВЕРШЕНО после {epochs_actually_run} эпох ---")
            if best_model_path_stage and Path(best_model_path_stage).exists():
                print(f"✅ Лучшая модель ({mode}) сохранена в: {Path(best_model_path_stage).name}")
                print(f"   Лучший Val Levenshtein ({mode}): {best_val_levenshtein:.4f}")
            else:
                # Если лучшая модель не сохранилась, но чекпоинт был (для finetune),
                # возможно, стоит вернуть исходный чекпоинт? Или ошибку?
                # Пока возвращаем пустой путь и inf.
                print(f"⚠️ Лучшая модель ({mode}) не была сохранена во время цикла.")
                if is_finetune_stage and absolute_checkpoint_path:
                     print(f"   (Исходный чекпоинт был: {Path(absolute_checkpoint_path).name})")
                best_model_path_stage = ""
                best_val_levenshtein = float('inf')

            # Логирование финальных метрик и артефакта в MLflow
            if IS_MLFLOW_ACTIVE and mlflow.active_run():
                try:
                    final_lev_metric = best_val_levenshtein if np.isfinite(best_val_levenshtein) else -1.0
                    mlflow.log_metric(f"final_best_{mode}_val_levenshtein", final_lev_metric)
                    mlflow.log_param(f"final_{mode}_epochs_run", epochs_actually_run)
                    final_model_name = Path(best_model_path_stage).name if best_model_path_stage else "N/A"
                    mlflow.log_param(f"final_best_{mode}_model_path", final_model_name)
                    # Логируем лучшую модель как артефакт
                    if best_model_path_stage and Path(best_model_path_stage).exists():
                         mlflow.log_artifact(best_model_path_stage, artifact_path=f"best_model_{mode}")
                         print("Артефакт лучшей модели этапа залогирован в MLflow.")
                except Exception as e_art_final:
                    print(f"Предупреждение: Ошибка логирования финальных артефактов/метрик в MLflow: {e_art_final}")
        else:
            print(f"\nЦикл обучения ({mode.upper()}) не был запущен или прерван до первой эпохи.")
            # Если чекпоинт был, возвращаем его
            if is_finetune_stage and absolute_checkpoint_path:
                 best_model_path_stage = absolute_checkpoint_path
                 # Пытаемся извлечь метрику из чекпоинта, если она там есть
                 try: best_val_levenshtein = checkpoint.get('val_levenshtein', float('inf'))
                 except NameError: best_val_levenshtein = float('inf') # checkpoint может быть не определен
                 print(f"Возвращается исходный чекпоинт: {Path(best_model_path_stage).name} (Lev: {best_val_levenshtein:.4f})")
            else:
                 best_model_path_stage = ""
                 best_val_levenshtein = float('inf')
            if IS_MLFLOW_ACTIVE and mlflow.active_run():
                with contextlib.suppress(Exception): mlflow.log_param(f"final_{mode}_epochs_run", 0)

    # ==========================================================================
    # Шаг 11: Обработка Ошибок и Очистка Ресурсов
    # ==========================================================================
    except KeyboardInterrupt:
        print(f"\n❗️ Обучение ({mode.upper()}) прервано пользователем (KeyboardInterrupt).")
        # Возвращаем то, что успели сохранить (или пустой путь)
        if not best_model_path_stage: best_val_levenshtein = float('inf')
    except ValueError as e_val: # Ловим ValueError (например, от DataLoader, Scheduler, Model Init)
        print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА (ValueError) во время '{mode}': {e_val}")
        traceback.print_exc(limit=2)
        best_model_path_stage = ""
        best_val_levenshtein = float('inf')
        if IS_MLFLOW_ACTIVE and mlflow.active_run():
             with contextlib.suppress(Exception): mlflow.set_tag("pipeline_error_stage", f"{mode}_value_error")
    except Exception as e:
        print(f"\n❌ НЕПРЕДВИДЕННАЯ КРИТИЧЕСКАЯ ОШИБКА ({type(e).__name__}) во время '{mode}':")
        traceback.print_exc()
        best_model_path_stage = ""
        best_val_levenshtein = float('inf')
        if IS_MLFLOW_ACTIVE and mlflow.active_run():
             with contextlib.suppress(Exception): mlflow.set_tag("pipeline_error_stage", f"{mode}_exception")
    finally:
        # --- Очистка ресурсов ---
        print(f"\nОчистка ресурсов после '{mode}'...")
        # Удаляем ссылки на большие объекты, чтобы помочь GC
        # Проверяем существование переменных перед удалением
        vars_to_delete = [
            'model', 'optimizer', 'criterion', 'scaler', 'scheduler',
            'train_loader', 'val_loader', 'train_ds', 'val_ds', 'spec_augmenter'
        ]
        for var_name in vars_to_delete:
            if var_name in locals():
                try:
                    del locals()[var_name]
                except NameError: # На всякий случай
                    pass

        collected_items = gc.collect()
        print(f"  GC собрал {collected_items} объектов.")

        # Используем 'device', определенную в начале функции run_training
        if 'device' in locals() and isinstance(device, torch.device) and device.type == 'cuda':
             if torch.cuda.is_available():
                 with contextlib.suppress(Exception): torch.cuda.empty_cache()
                 print("  Кэш CUDA очищен.")

        print("Очистка ресурсов завершена.")
        # --- Конец блока finally ---

    # --- Возвращаем результат ---
    print(f"\n>> Завершение run_training (mode='{mode}').")
    final_path_str = Path(best_model_path_stage).name if best_model_path_stage else "N/A"
    final_metric_str = f"{best_val_levenshtein:.4f}" if np.isfinite(best_val_levenshtein) else "inf"
    print(f"   Возвращаемый путь (лучшая модель этапа): '{final_path_str}'")
    print(f"   Возвращаемая метрика (лучший Val Levenshtein): {final_metric_str}")

    # Убедимся, что возвращаем пустой путь, если модель не существует
    final_path_return = best_model_path_stage if best_model_path_stage and Path(best_model_path_stage).exists() else ""
    final_metric_return = best_val_levenshtein if np.isfinite(best_val_levenshtein) else float('inf')

    return final_path_return, final_metric_return

# --- Конец файла src/training/pipeline.py ---