# Realiza o vínculo entre as interfaces e as implementações concretas
from app.usecases.process_text import ProcessTextUseCase
from app.usecases.interfaces import IProcessTextUseCase

def get_process_text_use_case() -> IProcessTextUseCase:
    # Aqui você cria a instância de ProcessTextUseCase com a dependência necessária
    return ProcessTextUseCase()
