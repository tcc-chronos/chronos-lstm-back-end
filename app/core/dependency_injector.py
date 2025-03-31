# Realiza o vínculo entre as interfaces e as implementações concretas
from app.usecases.process_text import ProcessTextUseCase
from app.usecases.train_model import TrainModelUseCase
from app.usecases.predict_model import PredictModelUseCase
from app.usecases.interfaces import IProcessTextUseCase, ITrainModelUseCase, IPredictModelUseCase

def get_process_text_use_case() -> IProcessTextUseCase:
    # Aqui você cria a instância de ProcessTextUseCase com a dependência necessária
    return ProcessTextUseCase()

def get_train_model_use_case() -> ITrainModelUseCase:
    return TrainModelUseCase()

def get_predict_model_use_case() -> IPredictModelUseCase:
    return PredictModelUseCase()
