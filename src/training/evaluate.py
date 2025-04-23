# src/training/evaluate.py
import torch
import Levenshtein # pip install python-Levenshtein
import numpy as np
from typing import List, Dict, Tuple
import traceback # Добавим для отладки

def ctc_greedy_decode(logits: torch.Tensor, int_to_char: Dict[int, str], blank_idx: int) -> List[str]:
    """
    Выполняет жадное CTC декодирование (Best Path Decoding).
    (Версия из Ячейки 10)
    """
    decoded_preds = []
    if logits.dim() != 3:
        error_msg = f"Ошибка (ctc_greedy_decode): Ожидается 3D тензор (B, T, V), получено {logits.shape}"
        print(error_msg)
        batch_size = logits.shape[0] if logits.dim() > 0 else 1
        return ["ERROR_DECODE_SHAPE"] * batch_size

    best_paths = torch.argmax(logits, dim=2)

    for path in best_paths.cpu().numpy():
        path_no_duplicates = [k for i, k in enumerate(path) if i == 0 or k != path[i-1]]
        path_no_blanks = [c for c in path_no_duplicates if c != blank_idx]
        decoded_text = "".join([int_to_char.get(c, '?') for c in path_no_blanks])
        decoded_preds.append(decoded_text)

    return decoded_preds

def calculate_levenshtein(
    preds: List[str],
    targets_padded: torch.Tensor,
    target_lengths: torch.Tensor,
    int_to_char: Dict[int, str],
    pad_idx: int,
    blank_idx: int
) -> Tuple[float, List[Tuple[str, str]]]:
    """
    Рассчитывает среднее расстояние Левенштейна.
    (Версия из Ячейки 10)
    """
    total_distance = 0.0
    count = 0
    decoded_pairs = []

    targets_padded_cpu = targets_padded.cpu()
    target_lengths_cpu = target_lengths.cpu().numpy()

    max_target_len_in_batch = targets_padded_cpu.size(1) if targets_padded_cpu.dim() > 1 else 0

    for i in range(len(preds)):
        pred_text = preds[i]
        target_text = "ERROR_TARGET_DECODE"

        try:
            # Проверка на случай, если target_lengths_cpu пустой или некорректный
            if i >= len(target_lengths_cpu):
                print(f"Ошибка (Levenshtein): Индекс {i} вне диапазона длин целей ({len(target_lengths_cpu)}).")
                target_text = "ERROR_LEN_INDEX"
                decoded_pairs.append((pred_text, target_text))
                continue

            length = target_lengths_cpu[i]

            if length <= 0:
                target_text = ""
            elif length > max_target_len_in_batch:
                 print(f"Ошибка (Levenshtein): Длина цели ({length}) > макс. длины в батче ({max_target_len_in_batch}) для индекса {i}.")
                 target_text = "ERROR_LEN_MISMATCH"
            else:
                target_indices = targets_padded_cpu[i, :length].numpy()
                target_text = "".join([
                    int_to_char.get(idx, '?') for idx in target_indices
                    if idx != pad_idx and idx != blank_idx
                ])

            if isinstance(pred_text, str) and isinstance(target_text, str):
                distance = Levenshtein.distance(pred_text, target_text)
                total_distance += distance
                count += 1
            else:
                 print(f"Предупреждение (Levenshtein): Некорректные типы: pred={type(pred_text)}, target={type(target_text)}")
                 pred_text = str(pred_text); target_text = str(target_text) # Пытаемся сохранить

            decoded_pairs.append((pred_text, target_text))

        except IndexError as e_idx:
             print(f"❌ Ошибка (IndexError) при декодировании цели {i}, длина {length}, форма targets {targets_padded_cpu.shape}: {e_idx}")
             decoded_pairs.append((pred_text, "ERROR_TARGET_DECODE_INDEX"))
        except Exception as e:
             print(f"❌ Неизвестная ошибка при декодировании/Levenshtein для индекса {i}: {e}")
             traceback.print_exc(limit=1)
             decoded_pairs.append((pred_text, "ERROR_UNKNOWN"))

    avg_distance = total_distance / count if count > 0 else float('inf')
    avg_distance = round(avg_distance, 4) if np.isfinite(avg_distance) else float('inf')

    return avg_distance, decoded_pairs