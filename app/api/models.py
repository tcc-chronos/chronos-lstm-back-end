# Define modelos para o retorno da API (como response schemas)
from pydantic import BaseModel
from typing import Literal, Optional
from typing import List


class TrainModelRequest(BaseModel):
    file_path: Optional[str] = "data.csv"
    rnn_type: Optional[Literal["lstm", "gru"]] = "lstm"
    column_data: Optional[str] = "urn:ngsi-ld:SPweather:001_TEMPERATURA_MAXIMA_NA_HORA_ANT_AUT_Celsius"
    window_size: Optional[int] = 60
    multi_feature: Optional[bool] = False
    epochs: Optional[int] = 50
    batch_size: Optional[int] = 16
    learning_rate: Optional[float] = 0.001
    dense_activation: Optional[str] = "relu"
    rnn_units: Optional[List[int]] = [128]
    dense_units: Optional[List[int]] = [64]
    dropout_rate: Optional[float] = 0.2
    early_stopping_patience: Optional[int] = 5

class TrainModelResponse(BaseModel):
    status: str
    training_time: float
    mean_squared_error: float
    mean_absolute_error: float
    root_mean_squared_error: float
    mean_absolute_percentage_error: float
    r_2_score: float
    accuracy: float
    best_train_loss: float
    best_val_loss: float

class PredictModelRequest(BaseModel):
    file_path: Optional[str] = "data.csv"
    column_data: Optional[str] = "urn:ngsi-ld:SPweather:001_TEMPERATURA_MAXIMA_NA_HORA_ANT_AUT_Celsius"
    window_size: Optional[int] = 60
    multi_feature: Optional[bool] = False
    n_steps_ahead: Optional[int] = 2
    model_path: Optional[str] = "trained_model.keras"
    
class PredictionResponse(BaseModel):
    status: str
    forecast: List[float]
    prediction_time: float