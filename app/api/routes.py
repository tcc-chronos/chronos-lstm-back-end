# Define as rotas da API
import time
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Path, Query
from app.api.models import AvailableFeaturesResponse, ModelInformationResponse, PreprocessingRequest, PreprocessingResponse, TrainModelRequest, TrainModelResponse, PredictRequest, PredictionResponse
from app.core.dependency_injector import get_available_features_use_case, get_data_pre_processing_use_case, get_model_information_use_case, get_train_model_use_case, get_predict_use_case
from app.core.exceptions import ProcessingError
from app.entities.train_model_config import TrainModelConfig
from app.usecases.interfaces import IAvailableFeaturesUseCase, IDataPreprocessingUseCase, IModelInformationUseCase, ITrainModelUseCase, IPredictUseCase
from app.utils.enums import ActivationFunction, str_to_enum

router = APIRouter()

@router.get("/health", tags=["Health Check"])
async def health_check():
    return {"status": "ok", "message": "API is running"}

@router.post("/preprocessing")
async def train(request: PreprocessingRequest, pre_processing_use_case: IDataPreprocessingUseCase = Depends(get_data_pre_processing_use_case)):
    try:
        start_time = time.time()

        pre_processing_use_case.execute(
            None,
            request.file_path, 
            request.column_data,
            request.window_size,
            request.multi_feature,
            save_data=True,
        )

        end_time = time.time()
        training_time = end_time - start_time

        return PreprocessingResponse(
            status="success",
            training_time=training_time
        )
    
    except ProcessingError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train")
async def train(request: TrainModelRequest, train_model_use_case: ITrainModelUseCase = Depends(get_train_model_use_case)):
    try:
        config = TrainModelConfig(
            rnn_type=request.rnn_type,
            epochs=request.epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            dense_activation=str_to_enum(ActivationFunction, request.dense_activation),
            rnn_units=request.rnn_units,
            dense_units=request.dense_units,
            dropout_rate=request.dropout_rate,
            early_stopping_patience=request.early_stopping_patience,
            bidirecional=request.bidirecional
        )
        
        mse, mae, rmse, mape, r2, best_train_loss, best_val_loss, training_time = train_model_use_case.execute(
            request.file_path, 
            request.column_data,
            request.window_size,
            request.multi_feature,
            config,
        )

        return TrainModelResponse(
            success=True,
            training_time=training_time,
            training_datetime=datetime.now(),
            mean_squared_error=mse, 
            mean_absolute_error=mae,
            root_mean_squared_error=rmse,
            mean_absolute_percentage_error=mape,
            r_2_score=r2,
            best_train_loss=best_train_loss,
            best_val_loss=best_val_loss
        )
    
    except ProcessingError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict")
async def predict(request: PredictRequest, predict_use_case: IPredictUseCase = Depends(get_predict_use_case)):
    try:
        start_time = time.time()
        
        real_values, forecast_values = predict_use_case.execute(
            request.file_path, 
            request.rnn_type,
            request.n_steps_ahead,
        )
        
        end_time = time.time()
        prediction_time = end_time - start_time
        
        if forecast_values is None:
            raise HTTPException(status_code=500, detail="Previsão não disponível.")
        
        return PredictionResponse(
            status="success",
            prediction_time=prediction_time,
            real_values=real_values,
            forecast_values=forecast_values 
        )
    
    except ProcessingError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model/{rnn_type}")
async def get_model_information(
    rnn_type: str = Path(..., description="Tipo da RNN (ex: LSTM, GRU, etc)"),
    model_info_use_case: IModelInformationUseCase = Depends(get_model_information_use_case)
):
    try:
        success, training_time, training_datetime, mae, rmse = model_info_use_case.execute(rnn_type)
        if success is False:
            raise HTTPException(status_code=404, detail=f"Modelo '{rnn_type}' não encontrado.")

        return ModelInformationResponse(
            success=True,
            training_time=training_time,
            training_datetime=training_datetime,
            mean_absolute_error=mae,
            root_mean_squared_error=rmse
        )

    except ProcessingError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/features", tags=["Utils"])
async def get_available_features(
    file_path: str = Query("train.csv",  description="Caminho para o arquivo CSV"),
    available_features_use_case: IAvailableFeaturesUseCase = Depends(get_available_features_use_case)
):
    try:
        features = available_features_use_case.execute(file_path)
        return AvailableFeaturesResponse(
            success=True,
            features=features
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Arquivo CSV não encontrado.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
