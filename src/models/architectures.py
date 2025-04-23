import torch
import torch.nn as nn
import math
import warnings
from typing import Dict

# --- Импорт строительных блоков ---
from .building_blocks import ResBlock

class MorseRecognizer(nn.Module):
    """
    Модель CRNN (Convolutional Recurrent Neural Network) для распознавания Морзе.
    (v10.1 - Вынесена в модуль).
    """
    def __init__(self, config: Dict):
        super().__init__()
        model_cfg = config.get("model", {})
        input_freq_dim = model_cfg.get("freq_dim")
        vocab_size = model_cfg.get("vocab_size")

        if input_freq_dim is None: raise ValueError("Ключ 'freq_dim' отсутствует в CONFIG['model']!")
        if vocab_size is None: raise ValueError("Ключ 'vocab_size' отсутствует!")

        input_channels = 1 # Моно спектрограмма

        # Параметры CNN
        cnn_block_channels = model_cfg.get("cnn_block_channels", [32, 64])
        cnn_kernels = model_cfg.get("cnn_kernel_size", [[3, 5], [3, 5]])
        cnn_strides = model_cfg.get("cnn_stride", [[2, 2], [2, 2]])
        use_se = model_cfg.get("cnn_use_se", True)
        se_reduction = model_cfg.get("cnn_se_reduction_ratio", 16)
        num_cnn_blocks = len(cnn_block_channels)

        if not (len(cnn_kernels) == num_cnn_blocks and len(cnn_strides) == num_cnn_blocks):
            raise ValueError("Длины cnn_block_channels, cnn_kernel_size и cnn_stride должны совпадать!")

        # Создание CNN слоев
        cnn_layers = []
        in_ch = input_channels
        self._time_reduction_factor = 1
        self._freq_reduction_factor = 1
        current_freq_dim = input_freq_dim

        print("\n--- Инициализация ResNet-SE CNN слоев (из architectures.py) ---")
        print(f"  Начальная F={input_freq_dim}, SE={use_se} (ratio={se_reduction if use_se else 'N/A'})")

        for i in range(num_cnn_blocks):
            out_ch = cnn_block_channels[i]
            kernel = tuple(cnn_kernels[i]) # Убедимся, что это кортеж
            stride = tuple(cnn_strides[i]) # Убедимся, что это кортеж

            cnn_layers.append(
                ResBlock(in_channels=in_ch, out_channels=out_ch,
                         kernel_size=kernel, stride=stride,
                         use_se=use_se, se_reduction_ratio=se_reduction)
            )
            print(f"  Block {i+1}: ResBlock({in_ch}, {out_ch}, k={kernel}, s={stride}, se={use_se})")
            in_ch = out_ch
            self._freq_reduction_factor *= stride[0]
            self._time_reduction_factor *= stride[1]
            current_freq_dim = math.ceil(current_freq_dim / stride[0])
            # print(f"    -> F~{current_freq_dim}, F_reduct={self._freq_reduction_factor}, T_reduct={self._time_reduction_factor}") # Для отладки

        self.cnn = nn.Sequential(*cnn_layers)
        print(f"CNN: Общий фактор сжатия: Время={self._time_reduction_factor}, Частота={self._freq_reduction_factor}")
        if self._time_reduction_factor <= 0 or current_freq_dim <= 0:
             warnings.warn("!!! Фактор сжатия или выходная частота CNN <= 0! Проверьте stride.")

        # Вычисление входа для RNN
        rnn_input_size = self._calculate_rnn_input_size(input_channels, input_freq_dim, config.get("device", "cpu"))

        if rnn_input_size <= 0:
             raise ValueError(f"Ошибка: Рассчитанный размер входа RNN ({rnn_input_size}) некорректен.")

        # RNN Слой (BiGRU)
        rnn_hidden_size = model_cfg.get("rnn_hidden_size", 128)
        rnn_layers = model_cfg.get("rnn_layers", 2)
        rnn_dropout = model_cfg.get("rnn_dropout", 0.2) if rnn_layers > 1 else 0.0
        self.rnn = nn.GRU(
            input_size=rnn_input_size, hidden_size=rnn_hidden_size,
            num_layers=rnn_layers, batch_first=True,
            bidirectional=True, dropout=rnn_dropout
        )
        print(f"--- Инициализация RNN ---")
        print(f"  BiGRU: Input={rnn_input_size}, Hidden={rnn_hidden_size}, Layers={rnn_layers}, Dropout={rnn_dropout:.2f}")

        # Классификатор
        classifier_dropout = model_cfg.get("dropout", 0.2)
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(rnn_hidden_size * 2, vocab_size)
        print(f"--- Инициализация Классификатора ---")
        print(f"  Dropout: {classifier_dropout:.2f}")
        print(f"  Linear: In={rnn_hidden_size * 2}, Out={vocab_size}")
        print("-" * 40)

    def _calculate_rnn_input_size(self, C_in: int, F_in: int, device: str) -> int:
        """ Вычисляет размер входа для RNN с помощью dummy forward. """
        rnn_input_size = 0
        try:
            self.eval() # Важно для BatchNorm и Dropout
            with torch.no_grad():
                # T должно быть достаточно большим
                dummy_T = max(100, self._time_reduction_factor * 5)
                dummy_input = torch.randn(1, C_in, dummy_T, F_in).to(device)
                # Перемещаем CNN на нужное устройство ПЕРЕД dummy forward
                cnn_temp = self.cnn.to(device)
                dummy_output = cnn_temp(dummy_input) # -> (B=1, C_out, T_red, F_red)
                # Возвращаем CNN на CPU (если он там был)
                self.cnn.to("cpu" if "cpu" in str(self.cnn[0].conv1.weight.device) else device)

                cnn_output_channels = dummy_output.shape[1]
                cnn_output_freq_dim = dummy_output.shape[3]
                rnn_input_size = cnn_output_channels * cnn_output_freq_dim
                # print(f"--- Расчет входа RNN (dummy forward) ---") # Отладочный вывод
                # print(f"  Dummy Input: {dummy_input.shape}")
                # print(f"  Dummy Output (CNN): {dummy_output.shape}")
                # print(f"  RNN Input Size (C_out * F_red): {cnn_output_channels} * {cnn_output_freq_dim} = {rnn_input_size}")
            self.train() # Возвращаем в режим train
        except Exception as e:
            print(f"!!! Ошибка при dummy forward для rnn_input_size: {e}")
            # Запасной вариант: ручной расчет
            model_cfg = self.config.get("model", {}) # Получаем конфиг снова
            cnn_output_channels = model_cfg.get("cnn_block_channels", [64])[-1]
            cnn_output_freq_dim = math.ceil(F_in / self._freq_reduction_factor)
            rnn_input_size = cnn_output_channels * cnn_output_freq_dim
            print(f"!!! Используется расчетный rnn_input_size = {rnn_input_size} !!!")
        return rnn_input_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Прямой проход: (B, C, T_in, F_in) -> (B, T_out, V) """
        # 1. CNN: (B, C, T_in, F_in) -> (B, C_out, T_red, F_red)
        x = self.cnn(x)
        # 2. Reshape для RNN: (B, T_red, C_out * F_red)
        B, C_out, T_red, F_red = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T_red, C_out * F_red)
        # 3. RNN: (B, T_red, Features) -> (B, T_red, RNN_hidden * 2)
        x, _ = self.rnn(x)
        # 4. Classifier: (B, T_red, RNN_hidden * 2) -> (B, T_red, VocabSize)
        x = self.dropout(x)
        x = self.classifier(x)
        # Возвращаем логиты (НЕ log_softmax)
        return x

    def get_time_reduction_factor(self) -> int:
        """ Возвращает общий фактор сжатия времени CNN. """
        return max(1, self._time_reduction_factor)