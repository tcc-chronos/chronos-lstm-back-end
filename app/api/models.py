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

class ProcessedTextResponse(BaseModel):
    status: str
    training_time: float
    data: Tuple
