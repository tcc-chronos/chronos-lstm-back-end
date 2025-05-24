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

            required_fields = [
                "training_time",
                "training_datetime",
                "mean_absolute_error",
                "root_mean_squared_error"
            ]

            for field in required_fields:
                if field not in metadata:
                    raise ProcessingError(f"Campo '{field}' ausente nos metadados.")

            return {
                "success": True,
                "training_time": metadata["training_time"],
                "training_datetime": metadata["training_datetime"],
                "mean_absolute_error": metadata["mean_absolute_error"],
                "root_mean_squared_error": metadata["root_mean_squared_error"],
                "train_config": {
                    key: value
                    for key, value in metadata.items()
                    if key not in required_fields
                }
            }

        except ProcessingError:
            raise
        except Exception as e:
            raise ProcessingError(f"Erro ao buscar informações do modelo: {str(e)}")
