from enum import Enum


class ActivationFunction(Enum):
    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    LINEAR = "linear"

class LossFunction(Enum):
    MSE = "mse"
    MAE = "mae"
    HUBER = "huber"

class OptimizerType(Enum):
    ADAM = "adam"
    SGD = "sgd"
    RMSPROP = "rmsprop"

def str_to_enum(enum_class, string_value):
    try:
        return enum_class(string_value)
    except ValueError:
        raise ValueError(f"Valor '{string_value}' não é válido para {enum_class.__name__}")
