from datetime import datetime
import os
import json
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from app.core.exceptions import ProcessingError
from app.infrastructure.csv_reader import CsvReader
from typing import List, Tuple
from app.usecases.interfaces import IPredictUseCase


class PredictUseCase(IPredictUseCase):
    def __init__(self):
        self.temp_dir = os.path.join(os.getcwd(), 'temp')

    def execute(self, file_path: str, rnn_type: str, n_steps_ahead: int) -> Tuple[List[Tuple[datetime, float]], List[Tuple[datetime, float]]]:
        metadata, model, x_scaler, y_scaler = self.load_model_and_metadata(rnn_type)
        df, future_df = self.load_and_validate_data(file_path, metadata)
        real_values, predicted_values = self.predict_historical(df, metadata, model, x_scaler, y_scaler)
        predicted_values = self.forecast_future(df, future_df, metadata, model, x_scaler, y_scaler, predicted_values, n_steps_ahead)
        return real_values, predicted_values

    def load_model_and_metadata(self, rnn_type: str):
        metadata_path = os.path.join(self.temp_dir, f"{rnn_type}_metadata.json")
        x_scaler_path = os.path.join(self.temp_dir, f"{rnn_type}_x_scaler.pkl")
        y_scaler_path = os.path.join(self.temp_dir, f"{rnn_type}_y_scaler.pkl")
        model_path = os.path.join(self.temp_dir, f"{rnn_type}.keras")

        for path in [metadata_path, x_scaler_path, y_scaler_path, model_path]:
            if not os.path.exists(path):
                raise ProcessingError(f"Modelo '{rnn_type}' não treinado. Arquivo não encontrado: {path}")

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        model = load_model(model_path)
        x_scaler = joblib.load(x_scaler_path)
        y_scaler = joblib.load(y_scaler_path)

        return metadata, model, x_scaler, y_scaler

    def load_and_validate_data(self, file_path: str, metadata: dict):
        df = CsvReader(file_path).read()
        timestamp_column = metadata.get('timestamp_column', 'timestamp')
        column_data = metadata['column_data']
        feature_columns = metadata.get('feature_columns')
        multi_feature = metadata['multi_feature']

        if not feature_columns:
            raise ProcessingError("Colunas de features não encontradas no metadata.")
        if column_data not in df.columns:
            raise ProcessingError(f"Coluna principal '{column_data}' não encontrada no CSV.")
        if timestamp_column not in df.columns:
            raise ProcessingError(f"Coluna de timestamp '{timestamp_column}' não encontrada no CSV.")
        for col in feature_columns:
            if col not in df.columns:
                raise ProcessingError(f"Coluna de feature '{col}' não encontrada no CSV.")

        df[timestamp_column] = pd.to_datetime(df[timestamp_column])

        future_df = None
        if multi_feature:
            future_path = os.path.join(os.path.dirname(file_path), 'real_future.csv')
            if not os.path.exists(future_path):
                raise ProcessingError("Arquivo 'real_future.csv' necessário para previsão multi-feature não encontrado.")

            future_df = CsvReader(future_path).read()
            if timestamp_column not in future_df.columns:
                raise ProcessingError(f"Coluna de timestamp '{timestamp_column}' não encontrada no CSV para previsão multi-feature.")
            for col in feature_columns:
                if col not in future_df.columns:
                    raise ProcessingError(f"Coluna '{col}' ausente no arquivo para previsão multi-feature.")

            future_df[timestamp_column] = pd.to_datetime(future_df[timestamp_column])
            future_df.sort_values(timestamp_column, inplace=True)

        return df, future_df

    def predict_historical(self, df: pd.DataFrame, metadata: dict, model, x_scaler, y_scaler):
        window_size = metadata['window_size']
        column_data = metadata['column_data']
        feature_columns = metadata['feature_columns']
        timestamp_column = metadata.get('timestamp_column', 'timestamp')

        real_values, predicted_values = [], []

        X = df[feature_columns].values
        X_scaled = x_scaler.transform(X)

        for i in range(window_size, len(df)):
            window_data = X_scaled[i - window_size:i]
            if np.any(np.isnan(window_data)):
                continue

            input_data = np.expand_dims(window_data, axis=0)
            prediction = model.predict(input_data, verbose=0)
            prediction_inverse = y_scaler.inverse_transform(prediction)

            timestamp = df.iloc[i][timestamp_column]
            real_value = df.iloc[i][column_data]

            if pd.notnull(real_value):
                try:
                    real_values.append((timestamp, float(real_value)))
                    predicted_values.append((timestamp, float(prediction_inverse[0][0])))
                except ValueError:
                    continue

        return real_values, predicted_values

    def forecast_future(self, df, future_df, metadata, model, x_scaler, y_scaler, predicted_values, n_steps_ahead):
        window_size = metadata['window_size']
        multi_feature = metadata['multi_feature']
        column_data = metadata['column_data']
        feature_columns = metadata['feature_columns']
        timestamp_column = metadata.get('timestamp_column', 'timestamp')

        last_window_data = x_scaler.transform(df[feature_columns].values)[-window_size:].copy()
        current_timestamp = df[timestamp_column].iloc[-1]
        freq = df[timestamp_column].diff().mode()[0]
        future_window = last_window_data

        for _ in range(n_steps_ahead):
            input_data = np.expand_dims(future_window, axis=0)
            prediction = model.predict(input_data, verbose=0)
            prediction_inverse = y_scaler.inverse_transform(prediction)
            current_timestamp += freq
            predicted_value = float(prediction_inverse[0][0])
            predicted_values.append((current_timestamp, predicted_value))

            if multi_feature:
                future_row = future_df[future_df[timestamp_column] == current_timestamp]
                if future_row.empty:
                    raise ProcessingError(f"Dados de entrada para timestamp {current_timestamp} não encontrados em 'real_future.csv'.")
                new_row = future_row[feature_columns].iloc[0].copy()
                new_row[feature_columns.index(column_data)] = prediction[0][0]
                new_row = np.array(new_row)
            else:
                new_row = np.array([prediction[0][0]])

            new_row_scaled = x_scaler.transform([new_row])[0]
            future_window = np.vstack([future_window[1:], new_row_scaled])

        return predicted_values
