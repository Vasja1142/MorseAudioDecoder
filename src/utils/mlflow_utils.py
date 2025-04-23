import mlflow
import sys
from typing import Dict

def setup_mlflow(config: Dict) -> bool:
    """
    Настраивает MLflow и возвращает статус активности (True/False).
    """
    # Проверяем, установлен ли MLflow
    if 'mlflow' not in sys.modules:
        print("MLflow не найден в окружении. Логирование будет неактивно.")
        return False

    # Проверяем наличие секции mlflow в конфиге
    if "mlflow" not in config or not isinstance(config.get("mlflow"), dict):
        print("Секция 'mlflow' отсутствует или некорректна в CONFIG. Логирование будет неактивно.")
        return False

    mlflow_config = config["mlflow"]
    # Используем .get для безопасного доступа к путям
    tracking_uri = config.get("paths", {}).get("mlflow_tracking_uri")
    experiment_name = mlflow_config.get("experiment_name")

    if not experiment_name:
        print("Имя эксперимента MLflow ('experiment_name') не указано. Логирование будет неактивно.")
        return False

    # Устанавливаем URI (если указан)
    if tracking_uri:
        try:
            mlflow.set_tracking_uri(tracking_uri)
            print(f"MLflow Tracking URI установлен: {tracking_uri}")
        except Exception as e_uri:
            print(f"❌ Ошибка установки MLflow Tracking URI ({tracking_uri}): {e_uri}. Логирование будет неактивно.")
            return False
    else:
        print("MLflow Tracking URI не указан, используется локальное логирование (папка 'mlruns').")

    # Пытаемся установить эксперимент и проверить соединение
    try:
        mlflow.set_experiment(experiment_name)
        print(f"MLflow Experiment '{experiment_name}' установлен.")
        # Проверка доступности URI (если указан)
        if tracking_uri:
             client = mlflow.tracking.MlflowClient()
             client.list_experiments() # Попытка выполнить запрос к серверу
             print("MLflow сервер доступен.")
        print("✅ MLflow успешно настроен и активен.")
        return True # Успех
    except Exception as e:
        print(f"❌ Ошибка настройки MLflow: {e}.")
        print("   Проверьте URI (если указан) и имя эксперимента.")
        print("   Логирование будет неактивно.")
        return False # Неудача