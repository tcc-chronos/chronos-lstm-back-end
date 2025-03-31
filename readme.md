# Chronos LSTM Back-end

Este repositório contém a implementação do back-end para o modelo LSTM do Chronos, responsável por processar e treinar modelos de previsão baseados em séries temporais.

## Configuração do Ambiente

Siga as etapas abaixo para configurar e executar o projeto corretamente:

### 1. Criar ambiente virtual
```sh
python -m venv venv
```

### 2. Ativar ambiente virtual
```sh
venv\Scripts\activate
```

### 3. Instalar dependências necessárias
```sh
pip install -r requirements.txt
```

### 4. Incluir o arquivo CSV na raiz do projeto
Certifique-se de incluir o arquivo de dados CSV necessário na raiz do projeto.

[Baixar processed_data.csv](https://github.com/tcc-chronos/impulse/blob/main/backend/processed_data.csv)

### 5. Executar o projeto
```sh
python -m uvicorn app.main:app --reload
```

### 6. Chamar o endpoint para treinamento
Use o seguinte comando `cURL` para iniciar o treinamento do modelo:
```sh
curl --location '{{CHRONOS_ENDPOINT}}/train' \
--header 'Content-Type: application/json' \
--data '{
    "file_path": "data.csv",
    "column_data": "urn:ngsi-ld:SPweather:001_TEMPERATURA_MAXIMA_NA_HORA_ANT_AUT_Celsius",
    "window_size": 60,
    "multi_feature": true,
    "epochs": 1,
    "batch_size": 16,
    "learning_rate": 0.001,
    "dense_activation": "relu",
    "loss_function": "mse",
    "optimizer": "adam",
    "num_lstm_layers": 1,
    "num_dense_layers": 0,
    "dropout_rate": 0.1,
    "early_stopping_patience": 5,
    "shuffle_data": true
}'
```

### 7. Desativar ambiente virtual (se necessário)
```sh
deactivate
```

## Tecnologias Utilizadas
- Python
- TensorFlow/Keras
- Scikit-learn
- Pandas
- FastAPI
- Uvicorn

## Manutenção
Se encontrar algum problema ou tiver sugestões de melhoria, fique à vontade para abrir uma issue ou enviar um pull request.

---

🚀 **Chronos LSTM Back-end - Prevendo o futuro, um dado por vez!**
