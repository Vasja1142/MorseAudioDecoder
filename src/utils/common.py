import random
import numpy as np
import torch
import os # Добавлен для бэкэндов

def set_seed(seed: int):
    """Устанавливает seed для воспроизводимости."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Для полной детерминированности можно раскомментировать, но может замедлить
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    # Попытка установить детерминизм для других бэкэндов, если необходимо
    # os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Seed установлен: {seed}")