# Define as rotas da API
from fastapi import APIRouter, Depends, HTTPException
from app.api.models import FilePathRequest, ProcessedTextResponse
from app.core.dependency_injector import get_process_text_use_case
from app.usecases.interfaces import IProcessTextUseCase

router = APIRouter()

@router.post("/process")
async def process_text(request: FilePathRequest, process_text_use_case: IProcessTextUseCase = Depends(get_process_text_use_case)):
    try:
        # Usando o caminho do arquivo fornecido no corpo da requisição
        processed_data = process_text_use_case.execute(request.file_path)
        return ProcessedTextResponse(status="success", data=processed_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
