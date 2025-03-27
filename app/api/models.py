# Define modelos para o retorno da API (como response schemas)
from pydantic import BaseModel

# Modelo para o corpo da requisição (Request)
class FilePathRequest(BaseModel):
    file_path: str

# Modelo para a resposta da API (Response)
class ProcessedTextResponse(BaseModel):
    status: str
    data: str
