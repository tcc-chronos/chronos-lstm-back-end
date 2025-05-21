import os
import json
from typing import Any, Dict
from app.core.exceptions import ProcessingError


class ModelInformationUseCase:
    def __init__(self):
        self.models_dir = os.path.join(os.getcwd(), "temp")

    def execute(self, rnn_type: str) -> Dict[str, Any]:
        try:
            metadata_path = os.path.join(self.models_dir, f"{rnn_type}_metadata.json")
            print(f"Lendo metadados de: {metadata_path}")

            if not os.path.exists(metadata_path):
                raise ProcessingError(f"Metadados do modelo '{rnn_type}' não encontrados.")

            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            # Validação campo a campo
            training_time = metadata.get("training_time")
            if training_time is None:
                raise ProcessingError("Campo 'training_time' ausente nos metadados.")

            training_datetime = metadata.get("training_datetime")
            if training_datetime is None:
                raise ProcessingError("Campo 'training_datetime' ausente nos metadados.")

            mean_absolute_error = metadata.get("mean_absolute_error")
            if mean_absolute_error is None:
                raise ProcessingError("Campo 'mean_absolute_error' ausente nos metadados.")

            root_mean_squared_error = metadata.get("root_mean_squared_error")
            if root_mean_squared_error is None:
                raise ProcessingError("Campo 'root_mean_squared_error' ausente nos metadados.")

            return (
                True,
                training_time,
                training_datetime,
                mean_absolute_error,
                root_mean_squared_error
            )

        except ProcessingError:
            raise
        except Exception as e:
            raise ProcessingError(f"Erro ao buscar informações do modelo: {str(e)}")
