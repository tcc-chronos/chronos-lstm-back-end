import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from typing import Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from app.core.exceptions import ProcessingError
from app.entities.train_model_config import TrainModelConfig
from app.infrastructure.csv_reader import CsvReader
from app.usecases.interfaces import ITrainModelUseCase

class TrainModelUseCase(ITrainModelUseCase):
    def __init__(self):
        pass

    def execute(self, 
            file_path: str, 
            column_data: str, 
            window_size: int, 
            multi_feature: bool,
            config: TrainModelConfig,
            model_save_path: str = "trained_model.h5"
        ) -> Tuple:
        # Validação das configurações
        self.validate_config(config)

        # Leitura dos dados
        df = CsvReader(file_path).read()
        
        # Preparação dos dados para teste
        x_train, x_test, y_train, y_test, y_scaler = self.data_preprocessing(df, column_data, window_size, multi_feature, config.shuffle_data)

        # Preparação do modelo
        model = self.model_compile(window_size, config, qtd_features=x_train.shape[2])

        # Treinamento do modelo
        metrics = self.model_train(model, multi_feature, x_train, x_test, y_train, y_test, y_scaler, config)

        # Salva o modelo em um arquivo .h5
        model.save(model_save_path)  

        # Retorno dos dados de treino
        return metrics

    def validate_config(self, config: TrainModelConfig):
        if config.num_lstm_layers <= 0:
            raise ProcessingError("Número de camadas LSTM deve ser maior que zero")
        if config.num_dense_layers < 0:
            raise ProcessingError("Número de camadas DENSE deve ser positivo")
        if config.dropout_rate < 0 or config.dropout_rate >= 1:
            raise ProcessingError("A desativação de neurônios (dropout_rate) deve estar entre [0, 1)")

    def data_preprocessing(self, df: pd.DataFrame, column_data: str, window_size: int, multi_feature: bool, shuffle_data: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
        if multi_feature:
            df = df.dropna().copy()
        else:
            df = df.dropna(subset=['timestamp', column_data]).copy()
        
        df['timestamp'] = pd.to_datetime(df['timestamp']).astype(np.int64) // 10**9
        x = df.drop(columns=[column_data]).values if multi_feature else df['timestamp'].values
        y = df[column_data].values

        x_scaler, y_scaler = StandardScaler(), StandardScaler()

        x_scaled = x_scaler.fit_transform(x) if multi_feature else x_scaler.fit_transform(x.reshape(-1, 1)).flatten()
        y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()

        x_seq, y_seq = [], []
        for i in range(len(x_scaled) - window_size):
            x_seq.append(x_scaled[i:i + window_size])
            y_seq.append(y_scaled[i + window_size])
        x_seq = np.array(x_seq) if multi_feature else np.array(x_seq).reshape(-1, window_size, 1)
        y_seq = np.array(y_seq)
        
        return (*train_test_split(x_seq, y_seq, test_size=0.2, random_state=42, shuffle=shuffle_data), y_scaler)


    def model_compile(self, window_size: int, config: TrainModelConfig, qtd_features: int = 1) -> Sequential:
        model = Sequential()
        model.add(Input(shape=(window_size, qtd_features)))
        for _ in range(config.num_lstm_layers):
            model.add(LSTM(128, return_sequences=True if _ < config.num_lstm_layers - 1 else False))
            model.add(Dropout(config.dropout_rate))
        for _ in range(config.num_dense_layers):
            model.add(Dense(64, activation=config.dense_activation.value))
        model.add(Dense(1))

        optimizer_instance = {"adam": Adam, "sgd": SGD, "rmsprop": RMSprop}[config.optimizer.value](learning_rate=config.learning_rate)
        model.compile(optimizer=optimizer_instance, loss=config.loss_function.value)
        
        return model

    def model_train(self, model: Sequential, multi_feature: bool, x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, y_scaler: StandardScaler, config: TrainModelConfig) -> Tuple:
        early_stop = EarlyStopping(monitor='val_loss', patience=config.early_stopping_patience, restore_best_weights=True)
        model.fit(
            x_train, y_train,
            epochs=config.epochs,
            batch_size=config.batch_size,
            validation_data=(x_test, y_test),
            callbacks=[early_stop],
            shuffle=config.shuffle_data, 
            verbose=1
        )
        
        predictions = model.predict(x_test).flatten()
        if not multi_feature:
            predictions = y_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
            y_test = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        r2 = r2_score(y_test, predictions)
        best_val_loss = min(model.history.history["val_loss"])

        return mse, mae, rmse, mape, r2, best_val_loss
