# Interface para leitura de dados (ex: arquivo txt)
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import StandardScaler
from app.entities.train_model_config import TrainModelConfig


class IDataPreprocessingUseCase(ABC):        
    @abstractmethod
    def execute(
        self,
        df: pd.DataFrame,
        file_path: str,
        column_data: str,
        window_size: int,
        multi_feature: bool,
        save_data: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, StandardScaler]:
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
    ) -> Tuple:
        """Executa o processamento de um arquivo csv e retorna os valores pós treino."""
        pass

class IPredictUseCase(ABC):
    @abstractmethod
    def execute(
        self, 
        file_path: str, 
        rnn_type: str, 
        n_steps_ahead: int,
    ) -> Tuple:
        """Executa a previsão com base nos dados mais recentes do arquivo CSV e retorna o valor previsto."""
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

class IModelInformationUseCase(ABC):        
    @abstractmethod
    def execute(
        self, 
        rnn_type: str
    ) -> Tuple:
        """Retorna as informações referentes ao modelo treinado."""
        pass
