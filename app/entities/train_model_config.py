from dataclasses import dataclass

from app.utils.enums import ActivationFunction, LossFunction, OptimizerType


@dataclass
class TrainModelConfig:
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 0.001
    dense_activation: ActivationFunction = ActivationFunction.RELU
    loss_function: LossFunction = LossFunction.MSE
    optimizer: OptimizerType = OptimizerType.ADAM
    num_lstm_layers: int = 1
    num_dense_layers: int = 1
    dropout_rate: float = 0.2
    early_stopping_patience: int = 5
    shuffle_data: bool = True
