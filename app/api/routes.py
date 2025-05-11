# Define as rotas da API
import time
from fastapi import APIRouter, Depends, HTTPException
from app.api.models import TrainModelRequest, TrainModelResponse, PredictModelRequest, PredictionResponse
from app.core.dependency_injector import get_train_model_use_case, get_predict_model_use_case
from app.entities.train_model_config import TrainModelConfig
from app.usecases.interfaces import ITrainModelUseCase, IPredictModelUseCase
from app.utils.enums import ActivationFunction, str_to_enum

router = APIRouter()

@router.get("/health", tags=["Health Check"])
async def health_check():
    return {"status": "ok", "message": "API is running"}

@router.post("/train")
async def train(request: TrainModelRequest, train_model_use_case: ITrainModelUseCase = Depends(get_train_model_use_case)):
    try:
        start_time = time.time()

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
        )
        
        mse, mae, rmse, mape, r2, accuracy, best_train_loss, best_val_loss = train_model_use_case.execute(
            request.file_path, 
            request.column_data,
            request.window_size,
            request.multi_feature,
            config,
        )

        end_time = time.time()
        training_time = end_time - start_time

        return TrainModelResponse(
            status="success",
            training_time=training_time,
            mean_squared_error=mse, 
            mean_absolute_error=mae,
            root_mean_squared_error=rmse,
            mean_absolute_percentage_error=mape,
            r_2_score=r2,
            accuracy=accuracy,
            best_train_loss=best_train_loss,
            best_val_loss=best_val_loss
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/predict")
async def predict(request: PredictModelRequest, predict_model_use_case: IPredictModelUseCase = Depends(get_predict_model_use_case)
):
    try:
        start_time = time.time()
        
        forecast = predict_model_use_case.execute(
            request.file_path, 
            request.column_data,
            request.window_size,
            request.multi_feature,
            request.n_steps_ahead,
            request.model_path
        )
        
        end_time = time.time()
        prediction_time = end_time - start_time
        
        if forecast is None:
            raise HTTPException(status_code=400, detail="Previsão não disponível.")
        
        return PredictionResponse(
            status="success",
            forecast=forecast,
            prediction_time=prediction_time
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))