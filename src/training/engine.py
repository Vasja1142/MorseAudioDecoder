# src/training/engine.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from tqdm.notebook import tqdm # Или from tqdm import tqdm для консоли
import numpy as np
import gc
import warnings
import math
import traceback
from typing import Optional, Tuple, List, Dict

# --- Импорты из других модулей src ---
from .evaluate import ctc_greedy_decode, calculate_levenshtein
# Предполагаем, что MorseRecognizer импортируется там, где вызывается train/evaluate

def train_epoch(
    model: nn.Module, dataloader: DataLoader, criterion: nn.CTCLoss,
    optimizer: optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: GradScaler, device: torch.device, config: dict,
    epoch_num: int, total_epochs: int, int_to_char: Dict[int, str],
    mode: str # 'train' или 'finetune'
) -> Tuple[float, float, float]:
    """
    Выполняет одну эпоху обучения модели.
    (Версия из Ячейки 11)
    """
    model.train()
    running_loss = 0.0
    total_lev_dist = 0.0
    total_lr = 0.0
    num_batches_processed = 0
    total_samples_processed = 0

    run_config = config.get(mode, config.get("training", {})) # Безопасный доступ
    ctc_cfg = config.get("ctc", {})
    use_amp = run_config.get("use_amp", False) and scaler.is_enabled()
    grad_accum_steps = run_config.get("gradient_accumulation_steps", 1)
    grad_clip_norm = run_config.get("gradient_clip_val", 1.0)
    blank_idx = ctc_cfg.get("blank_idx", 0)
    pad_idx = ctc_cfg.get("pad_idx", -1)

    try:
        time_factor = max(model.get_time_reduction_factor(), 1)
    except AttributeError:
        time_factor = 1
        warnings.warn("Не удалось получить time_reduction_factor из модели.")

    if len(dataloader) == 0:
        print(f"!!! ОШИБКА (train_epoch, {mode}): DataLoader пуст!")
        return float('inf'), float('inf'), 0.0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader),
                desc=f"Эпоха {epoch_num+1}/{total_epochs} [{mode.upper()}]",
                leave=False, ncols=1000)

    optimizer.zero_grad(set_to_none=True) # Оптимизация

    for batch_idx, batch_data in pbar:
        if batch_data is None: continue

        try:
            features, targets, feature_lengths, target_lengths = batch_data
            batch_size = features.size(0)
        except Exception as e_unpack:
             print(f"❌ Ошибка распаковки батча {batch_idx} ({mode}): {e_unpack}"); continue
        if batch_size == 0: continue

        try:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            feature_lengths_cpu = feature_lengths.cpu()
            target_lengths_cpu = target_lengths.cpu()
        except Exception as e_move:
            print(f"❌ Ошибка перемещения батча {batch_idx} ({mode}) на устройство: {e_move}"); continue

        input_lengths_ctc = torch.floor(feature_lengths_cpu.float() / time_factor + 1e-9).long().clamp(min=1)
        loss_value = float('nan'); lev_dist_batch = float('nan')
        current_lr = optimizer.param_groups[0]['lr']

        try:
            with autocast(enabled=use_amp):
                logits = model(features)
                log_probs = F.log_softmax(logits, dim=2).permute(1, 0, 2) # (T_out, B, V)

                output_length = log_probs.shape[0]
                if output_length == 0:
                    print(f"Предупреждение (Train Batch {batch_idx}, {mode}): Нулевая длина выхода! Пропуск."); continue

                input_lengths_ctc_clamped = input_lengths_ctc.clamp(max=output_length)
                target_lengths_clamped = target_lengths_cpu.clamp(max=targets.shape[1])

                valid_target_mask = target_lengths_clamped > 0
                if not torch.all(valid_target_mask):
                     # Фильтрация не нужна, CTC Loss умеет обрабатывать нулевые длины целей,
                     # если zero_infinity=True (установлено при создании criterion)
                     loss = criterion(log_probs, targets, input_lengths_ctc_clamped, target_lengths_clamped)
                else:
                     loss = criterion(log_probs, targets, input_lengths_ctc_clamped, target_lengths_clamped)

            if not torch.isfinite(loss):
                print(f"Предупреждение (Train Batch {batch_idx}, {mode}): NaN/Inf loss ({loss.item()})! Пропуск шага.")
                optimizer.zero_grad(set_to_none=True)
                loss_value = 30.0; lev_dist_batch = 30.0
                continue

            loss_value = loss.item()
            loss = loss / grad_accum_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(dataloader):
                if grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler and isinstance(scheduler, OneCycleLR):
                    scheduler.step()

            # Расчет Levenshtein (после шага, чтобы не мешать)
            # Используем импортированные функции
            decoded_preds = ctc_greedy_decode(logits.detach(), int_to_char, blank_idx)
            lev_dist_batch, _ = calculate_levenshtein(decoded_preds, targets, target_lengths_cpu, int_to_char, pad_idx, blank_idx)

            if not np.isfinite(lev_dist_batch): lev_dist_batch = 20.0

            running_loss += loss_value * batch_size
            total_lev_dist += lev_dist_batch * batch_size
            total_samples_processed += batch_size

        except RuntimeError as e:
             if "CUDA out of memory" in str(e):
                 print(f"❌ OOM ОШИБКА (Train Batch {batch_idx}, {mode})! Пропуск."); gc.collect(); torch.cuda.empty_cache(); optimizer.zero_grad(set_to_none=True); loss_value = float('nan'); lev_dist_batch = float('nan'); continue
             else: print(f"❌ RuntimeError (Train Batch {batch_idx}, {mode}): {e}"); traceback.print_exc(limit=1); loss_value = float('nan'); lev_dist_batch = float('nan')
        except Exception as e: print(f"❌ Ошибка ({type(e).__name__}) (Train Batch {batch_idx}, {mode}): {e}"); traceback.print_exc(limit=1); loss_value = float('nan'); lev_dist_batch = float('nan')

        total_lr += current_lr
        num_batches_processed += 1
        pbar.set_postfix(loss=f'{loss_value:.3f}' if np.isfinite(loss_value) else 'ERR', lev=f'{lev_dist_batch:.3f}' if np.isfinite(lev_dist_batch) else 'ERR', lr=f'{current_lr:.2e}')

        del features, targets, feature_lengths, target_lengths, logits, log_probs, loss
        if 'decoded_preds' in locals(): del decoded_preds

    pbar.close()

    if total_samples_processed == 0:
        print(f"!!! ПРЕДУПРЕЖДЕНИЕ (Train Epoch {epoch_num+1}, {mode}): Нет обработанных сэмплов!"); avg_loss = float('inf'); avg_lev = float('inf')
    else:
        avg_loss = running_loss / total_samples_processed; avg_lev = total_lev_dist / total_samples_processed
        avg_loss = round(avg_loss, 4) if np.isfinite(avg_loss) else float('inf'); avg_lev = round(avg_lev, 4) if np.isfinite(avg_lev) else float('inf')
    avg_lr = total_lr / num_batches_processed if num_batches_processed > 0 else 0.0

    gc.collect();
    if device == torch.device('cuda'): torch.cuda.empty_cache()
    return avg_loss, avg_lev, avg_lr


def evaluate_epoch(
    model: nn.Module, dataloader: DataLoader, criterion: nn.CTCLoss,
    device: torch.device, config: dict, int_to_char: Dict[int, str]
) -> Tuple[float, float, List[Tuple[str, str]]]:
    """
    Выполняет одну эпоху валидации модели.
    (Версия из Ячейки 11)
    """
    model.eval()
    running_loss = 0.0
    total_lev_dist = 0.0
    total_samples_processed = 0
    all_decoded_pairs = []

    base_train_cfg = config.get("training", {})
    ctc_cfg = config.get("ctc", {})
    use_amp = base_train_cfg.get("use_amp", False) # AMP для инференса
    blank_idx = ctc_cfg.get("blank_idx", 0)
    pad_idx = ctc_cfg.get("pad_idx", -1)

    try: time_factor = max(model.get_time_reduction_factor(), 1)
    except AttributeError: time_factor = 1; warnings.warn("Нет time_reduction_factor (evaluate).")

    if len(dataloader) == 0:
        print(f"!!! ОШИБКА (evaluate_epoch): Val DataLoader пуст!")
        return float('inf'), float('inf'), []

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="   [Валидация]", leave=False, ncols=1000)

    with torch.no_grad():
        for batch_idx_val, batch_data in pbar:
            if batch_data is None: continue

            try: features, targets, feature_lengths, target_lengths = batch_data; batch_size = features.size(0)
            except Exception as e_unpack: print(f"❌ Ошибка распаковки батча {batch_idx_val} (Val): {e_unpack}"); continue
            if batch_size == 0: continue

            try:
                features = features.to(device, non_blocking=True); targets = targets.to(device, non_blocking=True)
                feature_lengths_cpu = feature_lengths.cpu(); target_lengths_cpu = target_lengths.cpu()
            except Exception as e_move: print(f"❌ Ошибка перемещения батча {batch_idx_val} (Val) на устройство: {e_move}"); continue

            input_lengths_ctc = torch.floor(feature_lengths_cpu.float() / time_factor + 1e-9).long().clamp(min=1)
            loss_value = float('nan'); lev_dist_batch = float('nan'); decoded_pairs_batch = []

            try:
                with autocast(enabled=use_amp):
                    logits = model(features)
                    log_probs = F.log_softmax(logits, dim=2).permute(1, 0, 2)

                    output_length = log_probs.shape[0]
                    if output_length == 0: print(f"Предупреждение (Val Batch {batch_idx_val}): Нулевая длина выхода! Пропуск."); continue

                    input_lengths_ctc_clamped = input_lengths_ctc.clamp(max=output_length)
                    target_lengths_clamped = target_lengths_cpu.clamp(max=targets.shape[1])

                    # CTC Loss с zero_infinity=True обработает нулевые длины
                    loss = criterion(log_probs, targets, input_lengths_ctc_clamped, target_lengths_clamped)

                if torch.isfinite(loss): loss_value = loss.item()
                else: print(f"Предупреждение (Val Batch {batch_idx_val}): NaN/Inf loss!"); loss_value = 100.0

                # Используем импортированные функции
                decoded_preds = ctc_greedy_decode(logits, int_to_char, blank_idx)
                lev_dist_batch, decoded_pairs_batch = calculate_levenshtein(decoded_preds, targets, target_lengths_cpu, int_to_char, pad_idx, blank_idx)

                if not np.isfinite(lev_dist_batch): print(f"Предупреждение (Val Batch {batch_idx_val}): NaN/Inf Levenshtein!"); lev_dist_batch = 50.0

                running_loss += loss_value * batch_size
                total_lev_dist += lev_dist_batch * batch_size
                total_samples_processed += batch_size
                all_decoded_pairs.extend(decoded_pairs_batch)

            except RuntimeError as e:
                 if "CUDA out of memory" in str(e): print(f"❌ OOM ОШИБКА (ВАЛИДАЦИЯ, Батч {batch_idx_val})!"); gc.collect(); torch.cuda.empty_cache(); return float('inf'), float('inf'), [("OOM_ERROR", "OOM_ERROR")] * batch_size
                 else: print(f"❌ RuntimeError (Валидация, Batch {batch_idx_val}): {e}"); traceback.print_exc(limit=1); loss_value=float('nan'); lev_dist_batch=float('nan'); all_decoded_pairs.extend([("RUNTIME_ERROR","RUNTIME_ERROR")] * batch_size)
            except Exception as e: print(f"❌ Ошибка ({type(e).__name__}) (Валидация, Batch {batch_idx_val}): {e}"); traceback.print_exc(limit=1); loss_value=float('nan'); lev_dist_batch=float('nan'); all_decoded_pairs.extend([("UNKNOWN_ERROR","UNKNOWN_ERROR")] * batch_size)

            pbar.set_postfix(loss=f'{loss_value:.3f}' if np.isfinite(loss_value) else 'ERR', lev=f'{lev_dist_batch:.3f}' if np.isfinite(lev_dist_batch) else 'ERR')

            del features, targets, feature_lengths, target_lengths, logits, log_probs, loss
            if 'decoded_preds' in locals(): del decoded_preds
            # Безопасное удаление переменных
            vars_to_delete_batch = ['features', 'targets', 'feature_lengths', 'target_lengths',
                                    'logits', 'log_probs', 'loss', 'decoded_preds']
            current_locals = locals() # Получаем локальные переменные
            for var_name in vars_to_delete_batch:
                if var_name in current_locals:
                    try:
                        del current_locals[var_name] # Пытаемся удалить
                    except NameError: # На всякий случай
                        pass

    pbar.close()

    if total_samples_processed == 0:
        print("!!! ПРЕДУПРЕЖДЕНИЕ (evaluate_epoch): Нет обработанных сэмплов на валидации!"); avg_loss = float('inf'); avg_lev = float('inf')
    else:
        avg_loss = running_loss / total_samples_processed; avg_lev = total_lev_dist / total_samples_processed
        avg_loss = round(avg_loss, 4) if np.isfinite(avg_loss) else float('inf'); avg_lev = round(avg_lev, 4) if np.isfinite(avg_lev) else float('inf')

    print(f"   [Валидация завершена: Сэмплов = {total_samples_processed}]")
    gc.collect();
    if device == torch.device('cuda'): torch.cuda.empty_cache()
    return avg_loss, avg_lev, all_decoded_pairs