import pandas as pd
from app.infrastructure.csv_reader import CsvReader
from app.usecases.interfaces import ITrainModelUseCase
from app.entities.csv_data import CsvData
from typing import List

class TrainModelUseCase(ITrainModelUseCase):
    def __init__(self):
        pass

    def execute(self, file_path: str, column_data: str) -> List[str]:
        # Leitura dos dados
        df = CsvReader(file_path).read()
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['value'] = pd.to_numeric(df[column_data])
        
        csv_data_list = []
        for _, row in df.head(5).iterrows():
            csv_data = CsvData(row['timestamp'], row['value'])
            csv_data_list.append(str(csv_data))

        # Preparação dos dados para teste (Divisão X, y e afins)

        # Preparação do modelo

        # Treinamento do modelo

        # Retorno dos dados de treino
        
        return csv_data_list

