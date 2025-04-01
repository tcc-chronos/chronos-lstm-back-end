import pandas as pd

class CsvReader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def read(self):
        try:
            df = pd.read_csv(self.file_path)
            return df
        except FileNotFoundError:
            raise Exception(f"Arquivo {self.file_path} não encontrado")
        except UnicodeDecodeError:
            raise Exception(f"Erro ao decodificar o arquivo {self.file_path}. Verifique a codificação.")
        except Exception as e:
            raise Exception(f"Ocorreu um erro ao ler o arquivo {self.file_path}: {e}")

