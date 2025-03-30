import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error
from app.core.exceptions import ProcessingError
from app.infrastructure.csv_reader import CsvReader
from app.usecases.interfaces import ITrainModelUseCase
from typing import Tuple

class TrainModelUseCase(ITrainModelUseCase):
    def __init__(self):
        pass

    def execute(self, 
            file_path: str, 
            column_data: str, 
            window_size: int, 
            epochs: int = 50,
            batch_size: int = 16,
            learning_rate: float = 0.001
        ) -> Tuple:
        # Leitura dos dados
        df = CsvReader(file_path).read()
        
        # Preparação dos dados para teste
        split_data = self.data_preprocessing(df, column_data, window_size)

        # Preparação do modelo
        x_col_len = split_data[0].shape[2]
        model = self.model_compile(learning_rate, window_size, x_col_len)

        # Treinamento do modelo
        mse, mae = self.model_train(model, epochs, batch_size, split_data)

        # Retorno dos dados de treino
        return mse, mae
    
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

    def model_compile(self, learning_rate: float, window_size: int, x_col_len: int) -> Sequential:
        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape=(window_size, x_col_len)))
        model.add(LSTM(64))
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

        return model

    def model_train(self, model: Sequential, epochs: int, batch_size: int, split_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> Tuple[float, float]:
        early_stop = EarlyStopping(monitor='val_loss', patience=5)
        model.fit(
            split_data[0], split_data[2],
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(split_data[1], split_data[3]),
            callbacks=[early_stop],
            verbose=1
        )

        predictions = model.predict(split_data[1])
        mse = mean_squared_error(split_data[3], predictions)
        mae = mean_absolute_error(split_data[3], predictions)

        return mse, mae