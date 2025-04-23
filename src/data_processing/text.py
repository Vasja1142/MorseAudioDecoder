import pandas as pd
from typing import List, Dict, Tuple

def create_char_map(texts: List[str], ctc_config: dict) -> Tuple[Dict[str, int], Dict[int, str], int]:
    """
    Создает словари для преобразования символов в индексы и обратно.
    Учитывает blank и pad символы из конфигурации CTC.
    """
    blank_char = ctc_config.get("blank_char", "<blank>")
    pad_char = ctc_config.get("pad_char", "<pad>")
    blank_idx = ctc_config.get("blank_idx", 0)
    pad_idx = ctc_config.get("pad_idx", -1)

    # Фильтруем пустые/NaN значения и конвертируем в строки
    valid_texts = [str(text) for text in texts if pd.notna(text) and str(text) != '']
    num_invalid = len(texts) - len(valid_texts)
    if num_invalid > 0:
        print(f"Предупреждение: Обнаружено и проигнорировано {num_invalid} пустых/NaN значений в целевых текстах.")

    # Собираем уникальные символы
    unique_chars = set(char for text in valid_texts for char in text)
    sorted_chars = sorted(list(unique_chars))
    print(f"Найдено уникальных символов в текстах ({len(sorted_chars)}): {''.join(sorted_chars)}")

    # Проверяем конфликты с blank/pad
    if blank_char in sorted_chars:
        print(f"Предупреждение: Символ blank ('{blank_char}') найден в данных. Удален из словаря.")
        sorted_chars.remove(blank_char)
    if pad_char in sorted_chars:
        print(f"Предупреждение: Символ pad ('{pad_char}') найден в данных. Удален из словаря.")
        sorted_chars.remove(pad_char)

    # Создаем словарь char -> int
    char_to_int = {}
    current_idx = 0
    for char in sorted_chars:
        if current_idx == blank_idx:
            current_idx += 1 # Пропускаем индекс бланка
        char_to_int[char] = current_idx
        current_idx += 1

    # Добавляем бланк
    char_to_int[blank_char] = blank_idx

    # Создаем обратный словарь int -> char
    int_to_char = {i: char for char, i in char_to_int.items()}
    vocab_size = len(char_to_int) # Включая бланк

    # Проверки
    if len(int_to_char) != vocab_size:
         raise ValueError("Ошибка: Несоответствие размеров словарей char_to_int и int_to_char")
    if blank_idx not in int_to_char or int_to_char[blank_idx] != blank_char:
         raise ValueError(f"Ошибка: Проблема с символом blank (idx={blank_idx})")
    if pad_idx in int_to_char and pad_idx != blank_idx:
        raise ValueError(f"Ошибка: Индекс pad ({pad_idx}) конфликтует с символом '{int_to_char[pad_idx]}' (который не является бланком)")

    print(f"Размер словаря (Vocab Size, включая бланк): {vocab_size}")
    print(f"Словарь (idx: char): { {k: int_to_char[k] for k in sorted(int_to_char.keys())} }")
    print(f"Индекс Blank: {blank_idx}, Индекс Pad (для collate_fn): {pad_idx}")

    return char_to_int, int_to_char, vocab_size