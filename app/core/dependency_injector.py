# Realiza o vínculo entre as interfaces e as implementações concretas
from app.usecases.available_features import AvailableFeaturesUseCase
from app.usecases.data_preprocessing import DataPreprocessingUseCase
from app.usecases.model_information import ModelInformationUseCase
from app.usecases.predict import PredictUseCase
from app.usecases.train_model import TrainModelUseCase
from app.usecases.interfaces import IAvailableFeaturesUseCase, IDataPreprocessingUseCase, IModelInformationUseCase, IPredictUseCase, ITrainModelUseCase


def get_data_pre_processing_use_case() -> IDataPreprocessingUseCase:
    return DataPreprocessingUseCase()

def get_train_model_use_case() -> ITrainModelUseCase:
    return TrainModelUseCase()

def get_predict_use_case() -> IPredictUseCase:
    return PredictUseCase()

def get_model_information_use_case() -> IModelInformationUseCase:
    return ModelInformationUseCase()

def get_available_features_use_case() -> IAvailableFeaturesUseCase:
    return AvailableFeaturesUseCase()
