# Define modelos para o retorno da API (como response schemas)
from datetime import datetime
from pydantic import BaseModel
from typing import List, Literal, Optional, Tuple

class PreprocessingRequest(BaseModel):
    file_path: Optional[str] = "data.csv"
    column_data: Optional[str] = "urn:ngsi-ld:SPweather:001_TEMPERATURA_MAXIMA_NA_HORA_ANT_AUT_Celsius"
    window_size: Optional[int] = 60
    multi_feature: Optional[bool] = False

class PreprocessingResponse(BaseModel):
    status: str
    training_time: float

class TrainModelRequest(BaseModel):
    file_path: Optional[str] = "train.csv"
    rnn_type: Optional[Literal["lstm", "gru"]] = "lstm"
    column_data: Optional[str] = "urn:ngsi-ld:SPweather:001_TEMPERATURA_MAXIMA_NA_HORA_ANT_AUT_Celsius"
    window_size: Optional[int] = 60
    multi_feature: Optional[bool] = False
    epochs: Optional[int] = 5
    batch_size: Optional[int] = 16
    learning_rate: Optional[float] = 0.001
    dense_activation: Optional[Literal["relu", "sigmoid", "tanh", "linear"]] = "relu"
    rnn_units: Optional[List[int]] = [128]
    dense_units: Optional[List[int]] = [64]
    dropout_rate: Optional[float] = 0.2
    early_stopping_patience: Optional[int] = 5
    bidirecional: Optional[bool] = False

class TrainModelResponse(BaseModel):
    success: bool
    training_time: float
    training_datetime: datetime
    mean_squared_error: float
    mean_absolute_error: float
    root_mean_squared_error: float
    mean_absolute_percentage_error: float
    r_2_score: float
    best_train_loss: float
    best_val_loss: float

class PredictRequest(BaseModel):
    file_path: Optional[str] = "data.csv"
    rnn_type: Optional[Literal["lstm", "gru"]] = "lstm"
    n_steps_ahead: Optional[int] = 5
    
class PredictionResponse(BaseModel):
    status: str
    prediction_time: float
    real_values: List[Tuple[datetime, float]]
    forecast_values: List[Tuple[datetime, float]]

class ModelInformationResponse(BaseModel):
    success: bool
    training_time: float
    training_datetime: datetime
    mean_absolute_error: float
    root_mean_squared_error: float
