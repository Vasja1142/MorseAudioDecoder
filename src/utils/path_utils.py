from pathlib import Path

def create_full_path(filename: str, folder_path: Path) -> Path:
    """
    Формирует полный путь к аудиофайлу, гарантируя расширение .opus.
    Использует Pathlib для надежности.
    """
    # Убедимся, что filename - строка
    filename_str = str(filename)
    # Создаем Path объект из имени файла, меняем суффикс на .opus и объединяем с путем к папке
    return folder_path / Path(filename_str).with_suffix('.opus')