import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from app.core.exceptions import ProcessingError
from app.infrastructure.csv_reader import CsvReader
from app.usecases.interfaces import ITrainModelUseCase
from app.entities.csv_data import CsvData
from typing import Tuple

class TrainModelUseCase(ITrainModelUseCase):
    def __init__(self):
        pass

    def execute(self, file_path: str, column_data: str, window_size: int) -> Tuple:
        # Leitura dos dados
        df = CsvReader(file_path).read()
        
        # Preparação dos dados para teste (Divisão X, y)
        x_train, x_test, y_train, y_test = self.data_preprocessing(df, column_data, window_size)

        # Preparação do modelo

        # Treinamento do modelo

        # Retorno dos dados de treino

        return len(x_train), len(x_test), len(y_train), len(y_test)
        
    
    def data_preprocessing(self, df: pd.DataFrame, column_data: str, window_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x = pd.to_datetime(df['timestamp']).values
        y = pd.to_numeric(df[column_data]).values

        x_scaled = StandardScaler().fit_transform(x.reshape(-1, 1)).flatten()
        y_scaled = StandardScaler().fit_transform(y.reshape(-1, 1)).flatten()

        x_seq = []
        y_seq = []
        for i in range(len(x_scaled) - window_size):
            x_seq.append(x_scaled[i:i + window_size])
            y_seq.append(y_scaled[i + window_size])
        x_seq = np.array(x_seq)
        y_seq = np.array(y_seq)

        if len(x_seq) < 1:
            raise ProcessingError("Dados insuficientes para o treinamento após criar as sequências.")

        return train_test_split(x_seq, y_seq, test_size=0.2, random_state=42)


