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

            if len(feature_columns) > 1:
                new_row = future_window[-1].copy()
                new_row[feature_columns.index(column_data)] = prediction[0][0]
            else:
                new_row = np.array([prediction[0][0]])

            future_window = np.vstack([future_window[1:], new_row])

        return real_values, predicted_values

        # if multi_feature and not os.path.exists("csv_real_data.csv"):
        #     raise ProcessingError("Arquivo 'csv_real_data.csv' com dados futuros não encontrado.")
        
        # df_future = pd.read_csv("csv_real_data.csv")
        # df_future = df_future[feature_columns].reset_index(drop=True)

        # # Últimos dados do histórico
        # all_predictions = []
        # all_timestamps = []

        # # Frequência de tempo
        # freq = (df[timestamp_column].iloc[-1] - df[timestamp_column].iloc[-2]) if len(df) > 1 else timedelta(hours=1)
        # current_timestamp = df[timestamp_column].iloc[-1]

        # # Input inicial
        # if multi_feature:
        #     x_input = df[feature_columns].tail(window_size).values
        # else:
        #     if column_data not in df.columns:
        #         raise ProcessingError(f"Coluna '{column_data}' não encontrada no CSV.")
        #     x_input = df[[column_data]].tail(window_size).values

        # x_input_scaled = x_scaler.transform(x_input)
        # x_input_scaled = x_input_scaled.reshape(1, window_size, -1)

        # total_steps = len(df_future) + n_steps_ahead
        # real_values = df[[timestamp_column, column_data]].iloc[-len(df_future):].values.tolist()

        # for step in range(total_steps):
        #     pred_scaled = model.predict(x_input_scaled, verbose=0).flatten()[0]
        #     pred_value = y_scaler.inverse_transform([[pred_scaled]])[0][0]

        #     current_timestamp += freq
        #     all_predictions.append(pred_value)
        #     all_timestamps.append(current_timestamp)

        #     if step < len(df_future):
        #         if multi_feature:
        #             next_features = df_future.iloc[step].values.astype(float)
        #             feature_index = feature_columns.index(column_data)
        #             next_features[feature_index] = pred_scaled
        #             next_scaled = x_scaler.transform([next_features]).reshape(1, 1, -1)
        #         else:
        #             next_scaled = np.array([[[pred_scaled]]])
        #     else:
        #         # Autoregressivo puro
        #         if multi_feature:
        #             # Repete últimas features e substitui o target
        #             next_features = x_input_scaled[0, -1, :].copy()
        #             feature_index = feature_columns.index(column_data)
        #             next_features[feature_index] = pred_scaled
        #             next_scaled = np.array(next_features).reshape(1, 1, -1)
        #         else:
        #             next_scaled = np.array([[[pred_scaled]]])

        #     x_input_scaled = np.append(x_input_scaled[:, 1:, :], next_scaled, axis=1)

        # forecast_values = list(zip(all_timestamps, all_predictions))
        # real_values = [(df[timestamp_column].iloc[-len(df_future) + i], val) for i, (_, val) in enumerate(real_values)]

        # return real_values, forecast_values
