# Interface para leitura de dados (ex: arquivo txt)
from abc import ABC, abstractmethod
from typing import List

class IProcessTextUseCase(ABC):        
    @abstractmethod
    def execute(self, file_path: str) -> str:
        """Executa o processamento de um arquivo e retorna o texto processado."""
        pass

class ITrainModelUseCase(ABC):        
    @abstractmethod
    def execute(self, file_path: str, column_data: str) -> List[str]:
        """Executa o processamento de um arquivo csv e retorna os valores pós treino."""
        pass
