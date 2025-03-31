# Define as rotas da API
import time
from fastapi import APIRouter, Depends, HTTPException
from app.api.models import FilePathRequest, ProcessedTextResponse, TrainModelRequest, PredictionResponse, PredictModelRequest
from app.core.dependency_injector import get_process_text_use_case, get_train_model_use_case, get_predict_model_use_case
from app.entities.train_model_config import TrainModelConfig
from app.usecases.interfaces import IProcessTextUseCase, ITrainModelUseCase, IPredictModelUseCase
from app.utils.enums import ActivationFunction, LossFunction, OptimizerType, str_to_enum

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

        config = TrainModelConfig(
            epochs=request.epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            dense_activation=str_to_enum(ActivationFunction, request.dense_activation),
            loss_function=str_to_enum(LossFunction, request.loss_function),
            optimizer=str_to_enum(OptimizerType, request.optimizer),
            num_lstm_layers=request.num_lstm_layers,
            num_dense_layers=request.num_dense_layers,
            dropout_rate=request.dropout_rate,
            early_stopping_patience=request.early_stopping_patience,
            shuffle_data=request.shuffle_data
        )
        
        mse, mae, rmse, mape, r2, best_val_loss = train_model_use_case.execute(
            request.file_path, 
            request.column_data,
            request.window_size,
            request.multi_feature,
            config
        )

        end_time = time.time()
        training_time = end_time - start_time

        return ProcessedTextResponse(
            status="success",
            training_time=training_time,
            mean_squared_error=mse, 
            mean_absolute_error=mae,
            root_mean_squared_error=rmse,
            mean_absolute_percentage_error=mape,
            r_2_score=r2,
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
            model_path="trained_model.h5"
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