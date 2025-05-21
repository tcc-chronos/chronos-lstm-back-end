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


class PredictUseCase:
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
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])

        if not os.path.exists("csv_real_data.csv"):
            raise ProcessingError("Arquivo 'csv_real_data.csv' com dados futuros não encontrado.")
        
        df_future = pd.read_csv("csv_real_data.csv")
        df_future = df_future[feature_columns].reset_index(drop=True)

        x_scaler = joblib.load(x_scaler_path)
        y_scaler = joblib.load(y_scaler_path)

        model = load_model(model_path)

        # Últimos dados do histórico
        all_predictions = []
        all_timestamps = []

        # Frequência de tempo
        freq = (df[timestamp_column].iloc[-1] - df[timestamp_column].iloc[-2]) if len(df) > 1 else timedelta(hours=1)
        current_timestamp = df[timestamp_column].iloc[-1]

        # Input inicial
        if multi_feature:
            x_input = df[feature_columns].tail(window_size).values
        else:
            if column_data not in df.columns:
                raise ProcessingError(f"Coluna '{column_data}' não encontrada no CSV.")
            x_input = df[[column_data]].tail(window_size).values

        x_input_scaled = x_scaler.transform(x_input)
        x_input_scaled = x_input_scaled.reshape(1, window_size, -1)

        total_steps = len(df_future) + n_steps_ahead
        real_values = df[[timestamp_column, column_data]].iloc[-len(df_future):].values.tolist()

        for step in range(total_steps):
            pred_scaled = model.predict(x_input_scaled, verbose=0).flatten()[0]
            pred_value = y_scaler.inverse_transform([[pred_scaled]])[0][0]

            current_timestamp += freq
            all_predictions.append(pred_value)
            all_timestamps.append(current_timestamp)

            if step < len(df_future):
                if multi_feature:
                    next_features = df_future.iloc[step].values.astype(float)
                    feature_index = feature_columns.index(column_data)
                    next_features[feature_index] = pred_scaled
                    next_scaled = x_scaler.transform([next_features]).reshape(1, 1, -1)
                else:
                    next_scaled = np.array([[[pred_scaled]]])
            else:
                # Autoregressivo puro
                if multi_feature:
                    # Repete últimas features e substitui o target
                    next_features = x_input_scaled[0, -1, :].copy()
                    feature_index = feature_columns.index(column_data)
                    next_features[feature_index] = pred_scaled
                    next_scaled = np.array(next_features).reshape(1, 1, -1)
                else:
                    next_scaled = np.array([[[pred_scaled]]])

            x_input_scaled = np.append(x_input_scaled[:, 1:, :], next_scaled, axis=1)

        forecast_values = list(zip(all_timestamps, all_predictions))
        real_values = [(df[timestamp_column].iloc[-len(df_future) + i], val) for i, (_, val) in enumerate(real_values)]

        return real_values, forecast_values
