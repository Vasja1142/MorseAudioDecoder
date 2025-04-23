import re
from pathlib import Path
from typing import List

def find_available_perlin_maps(maps_dir: Path) -> List[int]:
    """
    Сканирует директорию maps_dir, находит существующие пары
    offset_map_k.npy и multi_map_k.npy и возвращает список
    доступных индексов k.
    """
    available_indices = []
    if not maps_dir.is_dir():
        print(f"Предупреждение: Директория карт Перлина не найдена: {maps_dir}")
        return []

    print(f"Сканирование директории карт Перлина: {maps_dir}")
    # Ищем файлы offset_map_????.npy
    for offset_path in maps_dir.glob("offset_map_*.npy"):
        match = re.search(r"offset_map_(\d+)\.npy", offset_path.name)
        if match:
            try:
                index = int(match.group(1))
                # Формируем ожидаемый путь к multi_map (с 4 знаками и нулями)
                multi_filename = f"multi_map_{index:04d}.npy"
                multi_path = maps_dir / multi_filename
                if multi_path.exists():
                    available_indices.append(index)
            except ValueError:
                print(f"Предупреждение: Не удалось извлечь числовой индекс из {offset_path.name}")
            except Exception as e:
                 print(f"Предупреждение: Ошибка при проверке пары для {offset_path.name}: {e}")

    if not available_indices:
        print("Предупреждение: Не найдено ни одной валидной пары карт Перлина.")
    else:
        num_found = len(available_indices)
        min_idx = min(available_indices)
        max_idx = max(available_indices)
        print(f"Найдено {num_found} доступных пар карт Перлина (индексы от {min_idx} до {max_idx}).")

    return sorted(list(set(available_indices))) # Сортируем и убираем дубликаты