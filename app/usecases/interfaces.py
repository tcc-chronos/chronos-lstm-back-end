# Interface para leitura de dados (ex: arquivo txt)
from abc import ABC, abstractmethod

class IProcessTextUseCase(ABC):        
    @abstractmethod
    def execute(self, file_path: str) -> str:
        """Executa o processamento de um arquivo e retorna o texto processado."""
        pass
