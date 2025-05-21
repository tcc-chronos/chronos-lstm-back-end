from app.core.exceptions import ProcessingError
from app.infrastructure.csv_reader import CsvReader
from app.usecases.interfaces import IAvailableFeaturesUseCase


class AvailableFeaturesUseCase(IAvailableFeaturesUseCase):
    def __init__(self):
        pass

    def execute(self, file_path: str) -> list[str]:
        try:
            df = CsvReader(file_path).read()
            features = [col for col in df.columns if col.lower() != "timestamp"]

            return features

        except ProcessingError:
            raise
        except Exception as e:
            raise ProcessingError(f"Erro ao buscar as features disponíveis: {str(e)}")
