import os
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from app.usecases.interfaces import IDataPreprocessingUseCase
from app.infrastructure.csv_reader import CsvReader


class DataPreprocessingUseCase(IDataPreprocessingUseCase):
    def __init__(self):
        pass

    def sort_by_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ordena o DataFrame pela coluna 'timestamp'."""
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.sort_values('timestamp')

    def handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')

        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        df[numeric_columns] = df[numeric_columns].interpolate(method='time')
        df[numeric_columns] = df[numeric_columns].fillna(method='bfill').fillna(method='ffill')

        if df[numeric_columns].isnull().any().any():
            raise ValueError("Ainda existem valores nulos após interpolação e preenchimento.")

        return df.reset_index()

    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trata os outliers nas colunas numéricas utilizando o método IQR (Interquartile Range)."""
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'timestamp' in numeric_columns:
            numeric_columns.remove('timestamp')

        for col in numeric_columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

        return df

    def convert_timestamp_to_seconds(self, df: pd.DataFrame) -> pd.DataFrame:
        """Converte a coluna 'timestamp' para inteiro (segundos)."""
        df['timestamp'] = df['timestamp'].astype(np.int64) // 10**9
        return df

    def create_sequences(self, x: np.ndarray, y: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Cria sequências de dados para o modelo, com base no tamanho da janela."""
        x_seq, y_seq = [], []
        for i in range(len(x) - window_size):
            x_seq.append(x[i:i + window_size])
            y_seq.append(y[i + window_size])

        x_seq = np.array(x_seq)
        y_seq = np.array(y_seq)

        return x_seq, y_seq

    def split_data(self, x_seq: np.ndarray, y_seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Divisão entre treino e teste."""
        return train_test_split(x_seq, y_seq, test_size=0.2, random_state=42, shuffle=False)

    def apply_scalers(self, x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
        """Aplica MinMaxScaler para normalizar as variáveis de entrada e alvo."""
        x_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()

        x_train_2d = x_train.reshape(-1, x_train.shape[-1])
        x_test_2d = x_test.reshape(-1, x_test.shape[-1])

        x_train_scaled = x_scaler.fit_transform(x_train_2d).reshape(x_train.shape)
        x_test_scaled = x_scaler.transform(x_test_2d).reshape(x_test.shape)

        y_train_scaled = y_scaler.fit_transform(y_train)
        y_test_scaled = y_scaler.transform(y_test)

        return x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled, x_scaler, y_scaler

    def save_data(self, x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled):
        # Define o caminho para o diretório 'temp/csvs'
        output_dir = 'temp/csvs'
        
        # Verifica se o diretório existe, caso contrário, cria-o
        os.makedirs(output_dir, exist_ok=True)
        
        # Salva os dados no diretório especificado
        pd.DataFrame(x_train_scaled.reshape(x_train_scaled.shape[0], -1)).to_csv(os.path.join(output_dir, 'x_train.csv'), index=False)
        pd.DataFrame(x_test_scaled.reshape(x_test_scaled.shape[0], -1)).to_csv(os.path.join(output_dir, 'x_test.csv'), index=False)
        pd.DataFrame(y_train_scaled).to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
        pd.DataFrame(y_test_scaled).to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)

    def execute(
        self,
        df: pd.DataFrame,
        file_path: str,
        column_data: str,
        window_size: int,
        multi_feature: bool,
        save_data: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
        """
        Função principal que orquestra o pré-processamento dos dados, realizando o tratamento de dados ausentes,
        outliers, conversão de timestamps, criação de sequências, divisão dos dados e aplicação dos scalers.
        """

        if df is None:
            df = CsvReader(file_path).read()

        df = self.sort_by_timestamp(df)

        df = self.handle_missing_data(df)

        # TODO: Validar impactos no modelo, uma vez que está removendo os valores 25% abaixo e 75% acima, talvez criando efeitos de flat inesperados
        # df = self.handle_outliers(df)

        # TODO: Avaliar se manter o timestamp ajuda, uma vez que os valores são MUITO grandes e podem tender a zero após o scaler
        # df = self.convert_timestamp_to_seconds(df)

        # Separar features e alvo
        if multi_feature:
            x = df.drop(columns=['timestamp']).values
        else:
            x = df[[column_data]].values
        y = df[[column_data]].values

        x_seq, y_seq = self.create_sequences(x, y, window_size)

        if not multi_feature:
            x_seq = x_seq.reshape(-1, window_size, 1)

        x_train, x_test, y_train, y_test = self.split_data(x_seq, y_seq)

        x_train, x_test, y_train, y_test, x_scaler, y_scaler = self.apply_scalers(
            x_train, x_test, y_train, y_test
        )

        if save_data:
            self.save_data(x_train, x_test, y_train, y_test)

        return x_train, x_test, y_train, y_test, x_scaler, y_scaler
