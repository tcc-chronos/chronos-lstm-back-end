import os
import json
import joblib
import numpy as np
from typing import Tuple
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from app.core.exceptions import ProcessingError
from app.entities.train_model_config import TrainModelConfig
from app.infrastructure.csv_reader import CsvReader
from app.usecases.data_preprocessing import DataPreprocessingUseCase
from app.usecases.interfaces import IDataPreprocessingUseCase, ITrainModelUseCase

class TrainModelUseCase(ITrainModelUseCase):
    def __init__(self):
        self.data_preprocessing_use_case: IDataPreprocessingUseCase = DataPreprocessingUseCase()

    def execute(self, 
            file_path: str, 
            column_data: str, 
            window_size: int, 
            multi_feature: bool,
            config: TrainModelConfig,
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

        self.save_model(config, column_data, window_size, multi_feature, model, x_scaler, y_scaler)

        # Retorno dos dados de treino
        return metrics

    def validate_config(self, config: TrainModelConfig):
        if not config.rnn_units or len(config.rnn_units) == 0:
            raise ProcessingError("Você deve especificar ao menos uma camada LSTM com seus neurônios.")
        if config.dense_units is None:
            raise ProcessingError("Você deve especificar a lista de unidades das camadas Dense (pode ser vazia).")
        if config.dropout_rate < 0 or config.dropout_rate >= 1:
            raise ProcessingError("A desativação de neurônios (dropout_rate) deve estar entre [0, 1).")


    def model_compile(self, window_size: int, config: TrainModelConfig, qtd_features: int = 1) -> Sequential:
        model = Sequential()
        model.add(Input(shape=(window_size, qtd_features)))

        RNNLayer = LSTM if config.rnn_type.lower() == "lstm" else GRU

        for i, units in enumerate(config.rnn_units):
            return_seq = i < len(config.rnn_units) - 1
            model.add(RNNLayer(
                units, 
                return_sequences=return_seq,
                dropout=config.dropout_rate,
                recurrent_dropout=config.dropout_rate
            ))
            model.add(Dropout(config.dropout_rate))

        for units in config.dense_units:
            model.add(Dense(units, activation=config.dense_activation.value))

        model.add(Dense(1))

        model.compile(optimizer=Adam(learning_rate=config.learning_rate), loss='mean_squared_error')
        return model


    def model_train(self, model: Sequential, multi_feature: bool, x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, y_scaler, config: TrainModelConfig) -> Tuple:
        early_stop = EarlyStopping(
            monitor='val_loss', 
            patience=config.early_stopping_patience, 
            restore_best_weights=True
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )

        history = model.fit(
            x_train, y_train,
            epochs=config.epochs,
            batch_size=config.batch_size,
            validation_data=(x_test, y_test),
            callbacks=[early_stop, reduce_lr],
            shuffle=False, 
            verbose=1
        )
        
        predictions = model.predict(x_test).flatten()
        if not multi_feature:
            predictions = y_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
            y_test = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        # Regressão - métricas clássicas
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mse)
        epsilon = 1e-10 
        mape = np.mean(np.abs((y_test - predictions) / (y_test + epsilon))) * 100
        r2 = r2_score(y_test, predictions)
        
        tolerance = 0.10 
        accuracy = np.mean(np.abs((y_test - predictions) / (y_test + epsilon)) < tolerance)

        best_epoch = np.argmin(history.history['val_loss'])
        best_val_loss = history.history['val_loss'][best_epoch]
        best_train_loss = history.history['loss'][best_epoch]

        return mse, mae, rmse, mape, r2, accuracy, best_train_loss, best_val_loss

    def save_model(self, 
            config:TrainModelConfig, 
            column_data: str, 
            window_size: int, 
            multi_feature: bool,
            model: Sequential, 
            x_scaler, 
            y_scaler
        ):
        # Salva o modelo em um arquivo .keras
        save_dir = os.path.join(os.getcwd(), 'temp')
        os.makedirs(save_dir, exist_ok=True)

        model.save(os.path.join(save_dir, f'{config.rnn_type}.keras'))
        joblib.dump(x_scaler, os.path.join(save_dir, f'{config.rnn_type}_x_scaler.pkl'))
        joblib.dump(y_scaler, os.path.join(save_dir, f'{config.rnn_type}_y_scaler.pkl'))

        metadata = {
            "rnn_type": config.rnn_type,
            "rnn_units": config.rnn_units,
            "dense_units": config.dense_units,
            "dropout_rate": config.dropout_rate,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "column_data": column_data,
            "window_size": window_size,
            "multi_feature": multi_feature,
        }

        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f"{config.rnn_type}_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)