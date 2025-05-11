# Interface para leitura de dados (ex: arquivo txt)
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler
from app.entities.train_model_config import TrainModelConfig


class IDataPreprocessingUseCase(ABC):        
    @abstractmethod
    def execute(
        self,
        df: pd.DataFrame,
        column_data: str,
        window_size: int,
        multi_feature: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
        """Executa o processamento de um arquivo csv e retorna os valores pós treino."""
        pass

class ITrainModelUseCase(ABC):        
    @abstractmethod
    def execute(
        self, 
        file_path: str, 
        column_data: str, 
        window_size: int, 
        multi_feature: bool, 
        config: TrainModelConfig,
        model_save_path: str,
    ) -> Tuple:
        """Executa o processamento de um arquivo csv e retorna os valores pós treino."""
        pass

class IPredictModelUseCase(ABC):
    @abstractmethod
    def execute(
        self, 
        file_path: str, 
        column_data: str, 
        window_size: int,
        multi_feature: bool,
        n_steps_ahead: int,
        model_path: str
    ) -> float:
        """Executa a previsão com base nos dados mais recentes do arquivo CSV e retorna o valor previsto."""
        pass