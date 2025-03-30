# Interface para leitura de dados (ex: arquivo txt)
from abc import ABC, abstractmethod
from typing import Tuple
from app.entities.train_model_config import TrainModelConfig


class IProcessTextUseCase(ABC):        
    @abstractmethod
    def execute(self, file_path: str) -> str:
        """Executa o processamento de um arquivo e retorna o texto processado."""
        pass

class ITrainModelUseCase(ABC):        
    @abstractmethod
    def execute(
        self, 
        file_path: str, 
        column_data: str, 
        window_size: int, 
        multi_feature: bool, 
        config: TrainModelConfig
    ) -> Tuple:
        """Executa o processamento de um arquivo csv e retorna os valores pós treino."""
        pass
