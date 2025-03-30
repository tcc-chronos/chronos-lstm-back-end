import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
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
            multi_feature: bool = False,
            epochs: int = 50,
            batch_size: int = 16,
            learning_rate: float = 0.001
        ) -> Tuple:
        # Leitura dos dados
        df = CsvReader(file_path).read()
        
        # Preparação dos dados para teste
        x_train, x_test, y_train, y_test, y_scaler = self.data_preprocessing(df, column_data, window_size, multi_feature)

        # Preparação do modelo
        model = self.model_compile(learning_rate, window_size, qtd_features=x_train.shape[2])

        # Treinamento do modelo
        mse, mae = self.model_train(model, epochs, batch_size, multi_feature, x_train, x_test, y_train, y_test, y_scaler)

        # Retorno dos dados de treino
        return mse, mae
    
    def data_preprocessing(self, df: pd.DataFrame, column_data: str, window_size: int, multi_feature: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, int]:
        if multi_feature:
            df = df.dropna().copy()
        else:
            df = df.dropna(subset=['timestamp', column_data]).copy()
        
        df['timestamp'] = pd.to_datetime(df['timestamp']).astype(np.int64) // 10**9

        x =  df.drop(columns=[column_data]).values if multi_feature else df['timestamp'].values
        y = df[column_data].values

        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        x_scaled = x_scaler.fit_transform(x) if multi_feature else x_scaler.fit_transform(x.reshape(-1, 1)).flatten()
        y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()

        x_seq, y_seq = [], []
        for i in range(len(x_scaled) - window_size):
            x_seq.append(x_scaled[i:i + window_size])
            y_seq.append(y_scaled[i + window_size])
        x_seq = np.array(x_seq) if multi_feature else np.array(x_seq).reshape(-1, window_size, 1)
        y_seq = np.array(y_seq)

        if len(x_seq) < 1:
            raise ProcessingError("Dados insuficientes para o treinamento após criar as sequências.")

        return (*train_test_split(x_seq, y_seq, test_size=0.2, random_state=42), y_scaler)

    def model_compile(self, learning_rate: float, window_size: int, qtd_features: int = 1) -> Sequential:
        model = Sequential([
            Input(shape=(window_size, qtd_features)),
            LSTM(128),
            Dense(64, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

        return model

    def model_train(self, model: Sequential, epochs: int, batch_size: int, multi_feature: bool, 
                    x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, 
                    y_test: np.ndarray, y_scaler: StandardScaler) -> Tuple[float, float]:
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, y_test),
            callbacks=[early_stop],
            verbose=1
        )

        # Previsões e métricas
        if multi_feature: 
            predictions = model.predict(x_test)
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
        else:
            predictions = model.predict(x_test).flatten()
            predictions = y_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
            y_test_inv = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

            mse = mean_squared_error(y_test_inv, predictions)
            mae = mean_absolute_error(y_test_inv, predictions)

        return mse, mae
