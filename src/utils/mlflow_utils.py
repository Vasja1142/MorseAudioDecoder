# src/utils/mlflow_utils.py

import mlflow
import sys
import contextlib
from typing import Dict, Any, Optional
from pathlib import Path # Добавим для проверки URI

def setup_mlflow(config: Dict[str, Any], project_root: Optional[Path] = None) -> bool:
    """
    Настраивает MLflow на основе конфигурации и возвращает статус активности.

    Args:
        config: Словарь конфигурации проекта.
        project_root: Корневой путь проекта (для разрешения относительных URI).

    Returns:
        True, если MLflow успешно настроен и активен, иначе False.
    """
    if 'mlflow' not in sys.modules:
        print("MLflow не найден в окружении. Логирование будет неактивно.")
        return False

    mlflow_config = config.get("mlflow")
    if not isinstance(mlflow_config, dict):
        print("Секция 'mlflow' отсутствует или некорректна в CONFIG. Логирование будет неактивно.")
        return False

    tracking_uri_rel = config.get("paths", {}).get("mlflow_tracking_uri")
    experiment_name = mlflow_config.get("experiment_name")

    if not experiment_name:
        print("Имя эксперимента MLflow ('experiment_name') не указано. Логирование будет неактивно.")
        return False

    # Обработка URI
    tracking_uri_abs = None
    if tracking_uri_rel:
        # Проверяем, является ли URI путем к файлу (для локального трекинга)
        uri_path = Path(tracking_uri_rel)
        if not uri_path.is_absolute():
            # Если путь относительный, делаем его абсолютным от корня проекта
            if project_root is None: project_root = Path.cwd()
            uri_path = (project_root / uri_path).resolve()
            # Для MLflow нужен формат file:///...
            tracking_uri_abs = uri_path.as_uri()
        else:
            # Если это уже абсолютный путь или http/databricks URI
            tracking_uri_abs = tracking_uri_rel if "://" in tracking_uri_rel else uri_path.as_uri()

        try:
            mlflow.set_tracking_uri(tracking_uri_abs)
            print(f"MLflow Tracking URI установлен: {tracking_uri_abs}")
        except Exception as e_uri:
            print(f"❌ Ошибка установки MLflow Tracking URI ({tracking_uri_abs}): {e_uri}. Логирование будет неактивно.")
            return False
    else:
        print("MLflow Tracking URI не указан, используется локальное логирование (папка 'mlruns').")

    # Установка эксперимента и проверка соединения
    try:
        mlflow.set_experiment(experiment_name)
        print(f"MLflow Experiment '{experiment_name}' установлен.")
        # Проверка доступности (только если URI был указан)
        if tracking_uri_abs:
             client = mlflow.tracking.MlflowClient()
             client.list_experiments(max_results=1) # Легкий запрос для проверки
             print("MLflow сервер/URI доступен.")
        print("✅ MLflow успешно настроен и активен.")
        return True
    except Exception as e:
        print(f"❌ Ошибка настройки MLflow: {e}.")
        print("   Проверьте URI (если указан) и имя эксперимента.")
        print("   Логирование будет неактивно.")
        return False

def log_mlflow_params(params: Dict[str, Any], prefix: Optional[str] = None):
    """
    Логирует параметры в MLflow, фильтруя неподдерживаемые типы и добавляя префикс.

    Args:
        params: Словарь параметров для логирования.
        prefix: Опциональный префикс для имен параметров.
    """
    if not mlflow.active_run():
        # print("Предупреждение: Попытка логирования параметров без активного MLflow run.")
        return # Не логируем, если нет активного запуска

    filtered_params: Dict[str, Any] = {}
    for k, v in params.items():
        # Логируем только базовые типы, которые MLflow точно поддерживает
        if isinstance(v, (str, int, float, bool)):
            key = f"{prefix}_{k}" if prefix else k
            # Ограничиваем длину строки, чтобы избежать ошибок MLflow
            filtered_params[key] = str(v)[:250] if isinstance(v, str) else v
        # Можно добавить обработку списков/кортежей базовых типов, если нужно
        # elif isinstance(v, (list, tuple)) and all(isinstance(i, (str, int, float, bool)) for i in v):
        #     key = f"{prefix}_{k}" if prefix else k
        #     filtered_params[key] = str(v)[:250] # Преобразуем в строку

    if filtered_params:
        with contextlib.suppress(Exception): # Подавляем ошибки логирования
            mlflow.log_params(filtered_params)
            # print(f"  MLflow: Залогированы параметры с префиксом '{prefix}': {list(filtered_params.keys())}")