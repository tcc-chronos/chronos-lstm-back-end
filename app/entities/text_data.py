# Define os modelos de dados que serão manipulados
class TextData:
    def __init__(self, content: str):
        self.content = content

    def __str__(self):
        return self.content
