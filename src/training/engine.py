# src/training/engine.py (Исправленная версия)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR
from tqdm.notebook import tqdm
import numpy as np
import gc
import warnings
import math
import traceback
from typing import Optional, Tuple, List, Dict, Any

# --- Импорты из других модулей src ---
try:
    from .evaluate import ctc_greedy_decode, calculate_levenshtein
except ImportError:
    print("КРИТИЧЕСКАЯ ОШИБКА: Не удалось импортировать функции из .evaluate в engine.py")
    # Заглушки
    def ctc_greedy_decode(logits, int_to_char, blank_idx): return ["DECODE_ERROR"] * logits.shape[1]
    def calculate_levenshtein(preds, targets, target_lengths, int_to_char, pad_idx, blank_idx): return 999.0, [("CALC_LEV_ERROR", "CALC_LEV_ERROR")] * len(preds)

def train_epoch(
    model: nn.Module, dataloader: DataLoader, criterion: nn.CTCLoss,
    optimizer: optim.Optimizer, scheduler: Optional[OneCycleLR], scaler: GradScaler,
    device: torch.device, config: Dict[str, Any], epoch_num: int, total_epochs: int,
    mode: str
) -> Tuple[float, float]:
    """
    Выполняет одну эпоху обучения модели. Возвращает средний лосс и средний LR (на начало батчей).
    """
    model.train()
    running_loss: float = 0.0
    total_lr: float = 0.0
    num_batches_processed: int = 0
    total_samples_processed: int = 0

    run_config = config.get(mode + 'ing', config.get("training", {}))
    use_amp = run_config.get("use_amp", False) and device.type == 'cuda' and scaler is not None and scaler.is_enabled()
    grad_accum_steps = max(1, run_config.get("gradient_accumulation_steps", 1))
    grad_clip_norm = run_config.get("gradient_clip_val", 1.0)
    blank_idx = config.get("ctc", {}).get("blank_idx", 0)

    try: time_factor = max(model.get_time_reduction_factor(), 1)
    except AttributeError: time_factor = 1

    if not dataloader: return float('inf'), 0.0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader),
                desc=f"Эпоха {epoch_num+1}/{total_epochs} [{mode.upper()}]",
                leave=False, ncols=1000)

    optimizer.zero_grad(set_to_none=True)

    # --- УДАЛЕН DEBUG PRINT ID ОПТИМИЗАТОРА/ШЕДУЛЕРА ---

    for batch_idx, batch_data in pbar:
        if batch_data is None: continue

        features, targets, feature_lengths, target_lengths = None, None, None, None
        logits, log_probs, loss = None, None, None
        loss_value = float('nan')
        current_lr_start_batch = optimizer.param_groups[0]['lr'] # LR в начале батча

        try:
            features, targets, feature_lengths, target_lengths = batch_data
            batch_size = features.size(0)
            if batch_size == 0: continue
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            feature_lengths_cpu = feature_lengths.cpu()
            target_lengths_cpu = target_lengths.cpu()

            with autocast(enabled=use_amp):
                logits = model(features)
                log_probs = F.log_softmax(logits, dim=2).permute(1, 0, 2)
                output_length = log_probs.shape[0]
                if output_length <= 0: continue
                input_lengths_ctc = torch.floor(feature_lengths_cpu.float() / time_factor + 1e-9).long().clamp(min=1, max=output_length)
                target_lengths_clamped = target_lengths_cpu.clamp(min=0, max=targets.shape[1])
                loss = criterion(log_probs, targets, input_lengths_ctc, target_lengths_clamped)

            if not torch.isfinite(loss):
                loss_value = 30.0
                optimizer.zero_grad(set_to_none=True)
                continue

            loss_value = loss.item()
            loss = loss / grad_accum_steps
            scaler.scale(loss).backward()

            # --- Шаг оптимизатора и планировщика ---
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(dataloader):
                # --- УДАЛЕНЫ DEBUG PRINT LR ДО/ПОСЛЕ SCHEDULER.STEP() ---
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler:
                    scheduler.step() # Вызов остается

            running_loss += loss_value * batch_size
            total_samples_processed += batch_size

        except RuntimeError as e:
             if "CUDA out of memory" in str(e): print(f"❌ OOM (Train {batch_idx})!"); gc.collect(); torch.cuda.empty_cache(); optimizer.zero_grad(set_to_none=True); loss_value = float('nan'); continue
             else: print(f"❌ RuntimeError (Train {batch_idx}): {e}"); loss_value = float('nan')
        except Exception as e: print(f"❌ Ошибка (Train {batch_idx}): {e}"); loss_value = float('nan')

        # --- Обновление TQDM ---
        total_lr += current_lr_start_batch
        num_batches_processed += 1
        pbar.set_postfix(loss=f'{loss_value:.3f}' if np.isfinite(loss_value) else 'ERR', lr=f'{current_lr_start_batch:.2e}')
        del features, targets, feature_lengths, target_lengths
        if 'feature_lengths_cpu' in locals(): del feature_lengths_cpu
        if 'target_lengths_cpu' in locals(): del target_lengths_cpu
        if 'input_lengths_ctc' in locals(): del input_lengths_ctc
        if 'target_lengths_clamped' in locals(): del target_lengths_clamped
        del logits, log_probs, loss

    pbar.close()
    avg_loss = (running_loss / total_samples_processed) if total_samples_processed > 0 else float('inf')
    avg_lr = (total_lr / num_batches_processed) if num_batches_processed > 0 else 0.0
    gc.collect(); torch.cuda.empty_cache()
    return round(avg_loss, 4) if np.isfinite(avg_loss) else float('inf'), avg_lr

def evaluate_epoch(
    model: nn.Module, dataloader: DataLoader, criterion: nn.CTCLoss,
    device: torch.device, config: Dict[str, Any], int_to_char: Dict[int, str]
) -> Tuple[float, float, List[Tuple[str, str]]]:
    """
    Выполняет одну эпоху валидации модели. Возвращает лосс, Levenshtein и примеры.
    """
    model.eval()
    running_loss: float = 0.0
    total_lev_dist: float = 0.0
    total_samples_processed: int = 0
    all_decoded_pairs: List[Tuple[str, str]] = []

    # Параметры из конфига
    base_train_cfg = config.get("training", {}) # Используем training конфиг для AMP
    ctc_cfg = config.get("ctc", {})
    use_amp = base_train_cfg.get("use_amp", False) and device.type == 'cuda'
    blank_idx = ctc_cfg.get("blank_idx", 0)
    pad_idx = ctc_cfg.get("pad_idx", -1)

    try: time_factor = max(model.get_time_reduction_factor(), 1)
    except AttributeError: time_factor = 1

    if not dataloader: return float('inf'), float('inf'), []

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="   [Валидация]", leave=False, ncols=1000)

    for batch_idx_val, batch_data in pbar:
        if batch_data is None: continue

        # Инициализация переменных для этого батча
        features, targets, feature_lengths, target_lengths = None, None, None, None
        logits, log_probs, loss = None, None, None # <-- Инициализация для блока del
        loss_value = float('nan'); lev_dist_batch = float('nan'); decoded_pairs_batch = []

        try:
            # 1. Распаковка и перемещение данных
            features, targets, feature_lengths, target_lengths = batch_data
            batch_size = features.size(0)
            if batch_size == 0: continue
            features = features.to(device, non_blocking=True) # (B, C, T, F) - уже 4D
            targets = targets.to(device, non_blocking=True)
            feature_lengths_cpu = feature_lengths.cpu()
            target_lengths_cpu = target_lengths.cpu()

            # 2. Прямой проход и вычисление Loss
            with autocast(enabled=use_amp):
                # model_input = features.unsqueeze(1) # <-- УБРАНО НЕПРАВИЛЬНОЕ unsqueeze
                logits = model(features) # (B, T_out, V) <-- Используем features напрямую
                log_probs = F.log_softmax(logits, dim=2).permute(1, 0, 2) # (T_out, B, V)

                output_length = log_probs.shape[0]
                if output_length <= 0: print(f"Предупреждение (Val {batch_idx_val}): Нулевой выход!"); continue

                # Вычисление длин для CTC Loss
                input_lengths_ctc = torch.floor(feature_lengths_cpu.float() / time_factor + 1e-9).long()
                input_lengths_ctc = input_lengths_ctc.clamp(min=1, max=output_length)
                target_lengths_clamped = target_lengths_cpu.clamp(min=0, max=targets.shape[1])

                loss = criterion(log_probs, targets, input_lengths_ctc, target_lengths_clamped)

            if torch.isfinite(loss): loss_value = loss.item()
            else: print(f"Предупреждение (Val {batch_idx_val}): NaN/Inf loss!"); loss_value = 100.0

            # 3. Декодирование и Levenshtein
            decoded_preds = ctc_greedy_decode(logits, int_to_char, blank_idx)
            lev_dist_batch, decoded_pairs_batch = calculate_levenshtein(
                decoded_preds, targets, target_lengths_cpu, int_to_char, pad_idx, blank_idx
            )
            if not np.isfinite(lev_dist_batch): print(f"Предупреждение (Val {batch_idx_val}): NaN/Inf Levenshtein!"); lev_dist_batch = 50.0

            running_loss += loss_value * batch_size
            total_lev_dist += lev_dist_batch * batch_size
            total_samples_processed += batch_size
            all_decoded_pairs.extend(decoded_pairs_batch)

        except RuntimeError as e:
             if "CUDA out of memory" in str(e): print(f"❌ OOM (ВАЛИДАЦИЯ {batch_idx_val})!"); gc.collect(); torch.cuda.empty_cache(); return float('inf'), float('inf'), [("OOM", "OOM")] * total_samples_processed # Прерываем валидацию при OOM
             else: print(f"❌ RuntimeError (Val {batch_idx_val}): {e}"); loss_value=float('nan'); lev_dist_batch=float('nan'); all_decoded_pairs.extend([("RUNTIME_ERR","RUNTIME_ERR")] * batch_size)
        except Exception as e: print(f"❌ Ошибка (Val {batch_idx_val}): {e}"); loss_value=float('nan'); lev_dist_batch=float('nan'); all_decoded_pairs.extend([("ERR","ERR")] * batch_size)

        # Обновление прогресс-бара и сборка мусора в конце итерации
        pbar.set_postfix(loss=f'{loss_value:.3f}' if np.isfinite(loss_value) else 'ERR', lev=f'{lev_dist_batch:.3f}' if np.isfinite(lev_dist_batch) else 'ERR')
        # Удаляем переменные батча
        del features, targets, feature_lengths, target_lengths
        if 'feature_lengths_cpu' in locals(): del feature_lengths_cpu
        if 'target_lengths_cpu' in locals(): del target_lengths_cpu
        if 'input_lengths_ctc' in locals(): del input_lengths_ctc
        if 'target_lengths_clamped' in locals(): del target_lengths_clamped
        # Удаляем переменные вычислений (теперь они инициализированы None)
        del logits, log_probs, loss
        if 'decoded_preds' in locals(): del decoded_preds

    pbar.close()
    avg_loss = (running_loss / total_samples_processed) if total_samples_processed > 0 else float('inf')
    avg_lev = (total_lev_dist / total_samples_processed) if total_samples_processed > 0 else float('inf')
    gc.collect(); torch.cuda.empty_cache()
    return round(avg_loss, 4) if np.isfinite(avg_loss) else float('inf'), round(avg_lev, 4) if np.isfinite(avg_lev) else float('inf'), all_decoded_pairs