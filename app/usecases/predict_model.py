from app.core.exceptions import ProcessingError
from app.infrastructure.csv_reader import CsvReader
from app.usecases.interfaces import IPredictModelUseCase
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanSquaredError 


class PredictModelUseCase(IPredictModelUseCase):
    def __init__(self):
        pass

    def execute(self, 
                file_path: str, 
                column_data: str, 
                window_size: int,
                model_path: str = "trained_model.h5"  # Caminho do modelo treinado
        ) -> float:
        """
        Executa o processamento de um arquivo CSV para fazer previsões com base no modelo LSTM.
        """
        # Leitura dos dados
        df = CsvReader(file_path).read()

        # Preparação dos dados para previsão
        x_scaled, y_scaler = self.data_preprocessing(df, column_data, window_size)

        # Carregar o modelo treinado a partir do arquivo
        model = load_model(model_path, custom_objects={'MeanSquaredError': MeanSquaredError})  # Carrega o modelo

        # Realiza a previsão
        forecast = self.model_predict(model, x_scaled, y_scaler)

        return forecast

    def data_preprocessing(self, df: pd.DataFrame, column_data: str, window_size: int) -> Tuple[np.ndarray, StandardScaler]:
        df = df.dropna(subset=['timestamp', column_data])
        x = pd.to_datetime(df['timestamp']).astype(np.int64) / 10**9 
        y = pd.to_numeric(df[column_data]).values 

        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

        x_scaled = x_scaler.fit_transform(x.values.reshape(-1, 1)).flatten()
        y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()

        x_seq, y_seq = [], []
        for i in range(len(x_scaled) - window_size):
            x_seq.append(x_scaled[i:i + window_size])
            y_seq.append(y_scaled[i + window_size])
        x_seq = np.array(x_seq).reshape(-1, window_size, 1)
        y_seq = np.array(y_seq)

        if len(x_seq) < 1:
            raise ProcessingError("Dados insuficientes para realizar a previsão após criar as sequências.")

        return x_seq, y_scaler

    
    def model_predict(self, model: Sequential, x_scaled: np.ndarray, y_scaler: StandardScaler) -> float:
        predictions = model.predict(x_scaled)
        forecast_value = y_scaler.inverse_transform(predictions).flatten()[0]

        return forecast_value