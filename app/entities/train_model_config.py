from dataclasses import dataclass
from typing import List
from app.utils.enums import ActivationFunction


@dataclass
class TrainModelConfig:
    rnn_type: str = "lstm"
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 0.001
    dense_activation: ActivationFunction = ActivationFunction.RELU
    rnn_units: List[int] = None
    dense_units: List[int] = None
    dropout_rate: float = 0.2
    early_stopping_patience: int = 5
    bidirecional: bool = False
