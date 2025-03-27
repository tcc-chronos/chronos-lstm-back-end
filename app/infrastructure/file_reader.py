# Leitura de arquivos (simula a conexão com base de dados)
from app.entities.text_data import TextData


class FileReader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def read(self):
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                return TextData(content)
        except FileNotFoundError:
            raise Exception(f"Arquivo {self.file_path} não encontrado")
        except UnicodeDecodeError:
            raise Exception(f"Erro ao decodificar o arquivo {self.file_path}. Verifique a codificação.")
