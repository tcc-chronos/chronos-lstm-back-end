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
        feature_columns = metadata.get('feature_columns')

        if not feature_columns:
            raise ProcessingError("Colunas de features não encontradas no metadata.")

        # CSV com dados históricos recentes
        df = CsvReader(file_path).read()

        # CSV com dados reais futuros para features
        if not os.path.exists("csv_real_data.csv"):
            raise ProcessingError("Arquivo 'csv_real_data.csv' com dados futuros não encontrado.")
        
        df_future = pd.read_csv("csv_real_data.csv")
        df_future = df_future[feature_columns].reset_index(drop=True)
        if df.shape[0] < window_size or df_future.shape[0] < n_steps_ahead:
            raise ProcessingError("Dados insuficientes para predição com base no tamanho da janela ou passos futuros.")

        # Preparar entrada inicial
        if multi_feature:
            x_input = df[feature_columns].tail(window_size).values
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

        for step in range(n_steps_ahead):
            next_pred = model.predict(x_input_scaled, verbose=0).flatten()[0]
            predictions.append(next_pred)

            if multi_feature:
                try:
                    # Obter as features reais do próximo tempo
                    real_next_features = df_future.iloc[step].values.astype(float)
                except Exception as e:
                    raise ProcessingError(f"Erro ao acessar features reais futuras: {e}")

                # Substituir o valor da feature alvo pela previsão
                feature_index = feature_columns.index(column_data)
                real_next_features[feature_index] = next_pred

                next_step_scaled = x_scaler.transform([real_next_features])  # shape (1, N)
                next_step_scaled = next_step_scaled.reshape(1, 1, -1)
            else:
                next_step_scaled = np.array([[[next_pred]]])

            x_input_scaled = np.append(x_input_scaled[:, 1:, :], next_step_scaled, axis=1)

        # Inverter normalização da variável alvo
        predictions = y_scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten().tolist()
        return predictions
