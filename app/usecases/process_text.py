# Lógica de processamento do texto
from app.infrastructure.file_reader import FileReader
from app.usecases.interfaces import IProcessTextUseCase

class ProcessTextUseCase(IProcessTextUseCase):
    def __init__(self):
        pass

    def execute(self, file_path: str):
        # Leitura do arquivo
        text_data = FileReader(file_path).read()
        
        # Processamento do conteúdo
        text_data.content = self.process_text(text_data.content)
        
        # Retorno do resultado processado
        return text_data.content

    def process_text(self, text: str):
        # Exemplo simples de processamento: converter texto para maiúsculas
        return text.upper()
