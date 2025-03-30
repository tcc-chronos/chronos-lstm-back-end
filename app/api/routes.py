# Define as rotas da API
import time
from fastapi import APIRouter, Depends, HTTPException
from app.api.models import FilePathRequest, ProcessedTextResponse, TrainModelRequest
from app.core.dependency_injector import get_process_text_use_case, get_train_model_use_case
from app.usecases.interfaces import IProcessTextUseCase, ITrainModelUseCase

router = APIRouter()

@router.post("/process")
async def process_text(request: FilePathRequest, process_text_use_case: IProcessTextUseCase = Depends(get_process_text_use_case)):
    try:
        # Usando o caminho do arquivo fornecido no corpo da requisição
        processed_data = process_text_use_case.execute(request.file_path)
        return ProcessedTextResponse(status="success", data=processed_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/train")
async def process_text(request: TrainModelRequest, train_model_use_case: ITrainModelUseCase = Depends(get_train_model_use_case)):
    try:
        start_time = time.time()
        
        return_data = train_model_use_case.execute(
            request.file_path, 
            request.column_data,
            request.window_size
        )

        end_time = time.time()
        training_time = end_time - start_time

        return ProcessedTextResponse(
            status="success",
            training_time=training_time,
            data=return_data
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
