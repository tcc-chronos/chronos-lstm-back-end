import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from typing import Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from app.core.exceptions import ProcessingError
from app.entities.train_model_config import TrainModelConfig
from app.infrastructure.csv_reader import CsvReader
from app.usecases.data_preprocessing import DataPreprocessingUseCase
from app.usecases.interfaces import IDataPreprocessingUseCase, ITrainModelUseCase

class TrainModelUseCase(ITrainModelUseCase):
    def __init__(self):
        # Injeção de dependência do DataPreprocessingUseCase
        self.data_preprocessing_use_case: IDataPreprocessingUseCase = DataPreprocessingUseCase()

    def execute(self, 
            file_path: str, 
            column_data: str, 
            window_size: int, 
            multi_feature: bool,
            config: TrainModelConfig,
            model_save_path: str
        ) -> Tuple:
        # Validação das configurações
        self.validate_config(config)

        # Leitura dos dados
        df = CsvReader(file_path).read()

        # Preparação dos dados para treino e teste utilizando o DataPreprocessingUseCase
        x_train, x_test, y_train, y_test, x_scaler, y_scaler = self.data_preprocessing_use_case.execute(
            df, column_data, window_size, multi_feature
        )

        # Preparação do modelo
        model = self.model_compile(window_size, config, qtd_features=x_train.shape[2])

        # Treinamento do modelo
        metrics = self.model_train(model, multi_feature, x_train, x_test, y_train, y_test, y_scaler, config)

        # Salva o modelo em um arquivo .h5
        model.save(model_save_path, save_format='keras')  # salva como .keras

        # Retorno dos dados de treino
        return metrics

    def validate_config(self, config: TrainModelConfig):
        if config.num_lstm_layers <= 0:
            raise ProcessingError("Número de camadas LSTM deve ser maior que zero")
        if config.num_dense_layers < 0:
            raise ProcessingError("Número de camadas DENSE deve ser positivo")
        if config.dropout_rate < 0 or config.dropout_rate >= 1:
            raise ProcessingError("A desativação de neurônios (dropout_rate) deve estar entre [0, 1)")

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

    def model_train(self, model: Sequential, multi_feature: bool, x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, y_scaler, config: TrainModelConfig) -> Tuple:
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
        epsilon = 1e-10 
        mape = np.mean(np.abs((y_test - predictions) / (y_test + epsilon))) * 100
        r2 = r2_score(y_test, predictions)
        best_val_loss = min(model.history.history["val_loss"])

        return mse, mae, rmse, mape, r2, best_val_loss
