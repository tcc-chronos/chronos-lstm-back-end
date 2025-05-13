# Realiza o vínculo entre as interfaces e as implementações concretas
from app.usecases.data_preprocessing import DataPreprocessingUseCase
from app.usecases.predict import PredictUseCase
from app.usecases.train_model import TrainModelUseCase
from app.usecases.predict_model import PredictModelUseCase
from app.usecases.interfaces import IDataPreprocessingUseCase, IPredictUseCase, ITrainModelUseCase, IPredictModelUseCase


def get_data_pre_processing_use_case() -> IDataPreprocessingUseCase:
    return DataPreprocessingUseCase()

def get_train_model_use_case() -> ITrainModelUseCase:
    return TrainModelUseCase()

def get_predict_use_case() -> IPredictUseCase:
    return PredictUseCase()

def get_predict_model_use_case() -> IPredictModelUseCase:
    return PredictModelUseCase()
