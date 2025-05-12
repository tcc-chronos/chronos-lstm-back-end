import os
import json
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from app.core.exceptions import ProcessingError
from app.infrastructure.csv_reader import CsvReader
from typing import List

class PredictUseCase:
    def __init__(self):
        self.temp_dir = os.path.join(os.getcwd(), 'temp')

    def execute(self, file_path: str, rnn_type: str, n_steps_ahead: int) -> List[float]:
        # Verificar arquivos necessários
        metadata_path = os.path.join(self.temp_dir, f"{rnn_type}_metadata.json")
        x_scaler_path = os.path.join(self.temp_dir, f"{rnn_type}_x_scaler.pkl")
        y_scaler_path = os.path.join(self.temp_dir, f"{rnn_type}_y_scaler.pkl")
        model_path = os.path.join(self.temp_dir, f"{rnn_type}.keras")

        for path in [metadata_path, x_scaler_path, y_scaler_path, model_path]:
            if not os.path.exists(path):
                raise ProcessingError(f"Modelo '{rnn_type}' não treinado. Arquivo não encontrado: {path}")

        # Carregar arquivos
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        window_size = metadata['window_size']
        multi_feature = metadata['multi_feature']
        column_data = metadata['column_data']

        df = CsvReader(file_path).read()

        if df.shape[0] < window_size:
            raise ProcessingError(f"Arquivo CSV deve conter pelo menos {window_size} linhas para predição.")

        # Selecionar os últimos `window_size` registros
        if multi_feature:
            x_input = df.tail(window_size).values
        else:
            if column_data not in df.columns:
                raise ProcessingError(f"Coluna '{column_data}' não encontrada no CSV.")
            x_input = df[[column_data]].tail(window_size).values

        x_scaler = joblib.load(x_scaler_path)
        y_scaler = joblib.load(y_scaler_path)

        x_input_scaled = x_scaler.transform(x_input)
        x_input_scaled = x_input_scaled.reshape(1, window_size, -1)

        model = load_model(model_path)

        predictions = []

        for _ in range(n_steps_ahead):
            next_pred = model.predict(x_input_scaled, verbose=0).flatten()[0]
            predictions.append(next_pred)

            # Atualizar input
            next_input = np.append(x_input_scaled[:, 1:, :], [[[next_pred]] if not multi_feature else [[next_pred]*x_input_scaled.shape[2]]], axis=1)
            x_input_scaled = next_input

        # Inversão da escala se necessário
        if not multi_feature:
            predictions = y_scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten().tolist()

        return predictions
