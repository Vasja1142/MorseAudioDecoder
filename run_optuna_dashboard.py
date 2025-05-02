import os
import subprocess
from pathlib import Path
import sys
# --- ДОБАВЛЕН ИМПОРТ ---
from typing import Optional 
# -----------------------

# --- Настройки ---
# Укажите путь к папке, где хранятся ваши .db файлы Optuna
DEFAULT_DB_DIR = Path("./optuna_databases")
# --- Конец Настроек ---

def find_db_files(db_dir: Path) -> list[Path]:
    """Находит все файлы .db в указанной директории."""
    if not db_dir.is_dir():
        print(f"Ошибка: Папка '{db_dir}' не найдена.")
        return []
    db_files = sorted(list(db_dir.glob("*.db")))
    return db_files

# Теперь Optional известен благодаря импорту
def choose_db_file(db_files: list[Path]) -> Optional[Path]:
    """Отображает список файлов и просит пользователя выбрать один."""
    if not db_files:
        print(f"Ошибка: В папке не найдено файлов .db")
        return None

    print("\nНайденные базы данных Optuna:")
    for i, db_path in enumerate(db_files):
        print(f"  {i + 1}: {db_path.name}")

    while True:
        try:
            choice_str = input(f"Введите номер БД для запуска дашборда (1-{len(db_files)}) или 'q' для выхода: ")
            if choice_str.lower() == 'q':
                print("Выход.")
                return None
            
            choice_idx = int(choice_str) - 1
            
            if 0 <= choice_idx < len(db_files):
                return db_files[choice_idx]
            else:
                print("Неверный номер. Пожалуйста, выберите номер из списка.")
        except ValueError:
            print("Неверный ввод. Пожалуйста, введите число или 'q'.")
        except KeyboardInterrupt:
            print("\nВыход по запросу пользователя.")
            return None

def launch_dashboard(db_path: Path):
    """Запускает optuna-dashboard для выбранного файла БД."""
    db_uri = f"sqlite:///{db_path.resolve()}"
    command = ["optuna-dashboard", db_uri]
    
    print(f"\nЗапуск команды: {' '.join(command)}")
    print("Нажмите Ctrl+C в этом окне, чтобы остановить дашборд...")
    
    process = None # Инициализируем переменную process
    try:
        process = subprocess.Popen(command)
        process.wait()
    except FileNotFoundError:
        print("\nОшибка: Команда 'optuna-dashboard' не найдена.")
        print("Убедитесь, что optuna-dashboard установлен (`pip install optuna-dashboard`)")
        print("и доступен в системном PATH.")
    except KeyboardInterrupt:
        print("\nПолучен сигнал KeyboardInterrupt. Завершение...")
        if process and process.poll() is None: 
             process.terminate()
             try:
                  process.wait(timeout=5)
             except subprocess.TimeoutExpired:
                  process.kill()
        print("Дашборд остановлен.")
    except Exception as e:
        print(f"\nПроизошла ошибка при запуске optuna-dashboard: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        db_directory = Path(sys.argv[1])
        print(f"Используется папка из аргумента командной строки: {db_directory}")
    else:
        db_directory = DEFAULT_DB_DIR
        print(f"Используется папка по умолчанию: {db_directory}")

    databases = find_db_files(db_directory)
    
    if databases:
        selected_file = choose_db_file(databases)
        if selected_file:
            launch_dashboard(selected_file)

    print("\nСкрипт завершил работу.")