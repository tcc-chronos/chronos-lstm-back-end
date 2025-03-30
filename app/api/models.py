# Define modelos para o retorno da API (como response schemas)
from pydantic import BaseModel
from typing import Tuple, Optional

# Modelo para o corpo da requisição (Request)
class FilePathRequest(BaseModel):
    file_path: str

# Modelo para a resposta da API (Response)
class ProcessedTextResponse(BaseModel):
    status: str
    data: str

class TrainModelRequest(BaseModel):
    file_path: Optional[str] = "data.csv"
    column_data: Optional[str] = "urn:ngsi-ld:SPweather:001_TEMPERATURA_MAXIMA_NA_HORA_ANT_AUT_Celsius"
    window_size: Optional[int] = 60
    multi_feature: Optional[bool] = False
    
    epochs: Optional[int] = 50
    batch_size: Optional[int] = 16
    learning_rate: Optional[float] = 0.001
    dense_activation: Optional[str] = "relu"
    loss_function: Optional[str] = "mse"
    optimizer: Optional[str] = "adam"
    num_lstm_layers: Optional[int] = 1
    num_dense_layers: Optional[int] = 1
    dropout_rate: Optional[float] = 0.2
    early_stopping_patience: Optional[int] = 5
    shuffle_data: Optional[bool] = True

class ProcessedTextResponse(BaseModel):
    status: str
    training_time: float
    mean_squared_error: float
    mean_absolute_error: float
