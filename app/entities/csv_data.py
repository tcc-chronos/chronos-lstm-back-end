from datetime import datetime

class CsvData:
    def __init__(self, timestamp: datetime, value: float):
        self.timestamp = timestamp
        self.value = value
    
    def __str__(self):
        return f'{self.timestamp} - {self.value}'
