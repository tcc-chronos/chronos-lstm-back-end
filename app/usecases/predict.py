from datetime import datetime, timedelta
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
        metadata_path = os.path.join(self.temp_dir, f"{rnn_type}_metadata.json")
        x_scaler_path = os.path.join(self.temp_dir, f"{rnn_type}_x_scaler.pkl")
        y_scaler_path = os.path.join(self.temp_dir, f"{rnn_type}_y_scaler.pkl")
        model_path = os.path.join(self.temp_dir, f"{rnn_type}.keras")

        for path in [metadata_path, x_scaler_path, y_scaler_path, model_path]:
            if not os.path.exists(path):
                raise ProcessingError(f"Modelo '{rnn_type}' não treinado. Arquivo não encontrado: {path}")

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        window_size = metadata['window_size']
        multi_feature = metadata['multi_feature']
        column_data = metadata['column_data']
        feature_columns = metadata.get('feature_columns')
        timestamp_column = metadata.get('timestamp_column', 'timestamp')

        if not feature_columns:
            raise ProcessingError("Colunas de features não encontradas no metadata.")

        df = CsvReader(file_path).read()
        if column_data not in df.columns:
            raise ProcessingError(f"Coluna principal '{column_data}' não encontrada no CSV.")
        elif timestamp_column not in df.columns:
            raise ProcessingError(f"Coluna de timestamp '{timestamp_column}' não encontrada no CSV.")
        else:
            for feature_column in feature_columns:
                if feature_column not in df.columns:
                    raise ProcessingError(f"Coluna de feature '{feature_column}' não encontrada no CSV.")

        df[timestamp_column] = pd.to_datetime(df[timestamp_column])

        future_df = None
        if multi_feature:
            future_path = os.path.join(os.path.dirname(file_path), 'real_future.csv')
            if not os.path.exists(future_path):
                raise ProcessingError("Arquivo 'real_future.csv' necessário para previsão multi-feature não encontrado.")

            future_df = CsvReader(future_path).read()

            if timestamp_column not in future_df.columns:
                raise ProcessingError(f"Coluna de timestamp '{timestamp_column}' não encontrada no CSV para previsão multi-feature.")
            
            future_df[timestamp_column] = pd.to_datetime(future_df[timestamp_column])
            future_df.sort_values(timestamp_column, inplace=True)
            
            for feature in feature_columns:
                if feature not in future_df.columns:
                    raise ProcessingError(f"Coluna '{feature}' ausente no arquivo para previsão multi-feature.")

        x_scaler = joblib.load(x_scaler_path)
        y_scaler = joblib.load(y_scaler_path)

        model = load_model(model_path)

        real_values = []
        predicted_values = []
        
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
        
        last_window_data = X_scaled[-window_size:].copy()
        last_timestamp = df.iloc[-1][timestamp_column]
        freq = df[timestamp_column].diff().mode()[0]

        future_window = last_window_data
        current_timestamp = last_timestamp

        for step in range(n_steps_ahead):
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

        return real_values, predicted_values

