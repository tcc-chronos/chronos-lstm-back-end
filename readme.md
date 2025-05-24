
# Chronos Back-end
Este repositório contém a implementação do back-end para o **Chronos**, um sistema de previsão baseado em redes neurais recorrentes (RNNs) como LSTM e GRU, treinadas com dados de séries temporais.

## 📦 Configuração do Ambiente
Siga os passos abaixo para executar o projeto localmente:

### 1. Criar ambiente virtual
```
python  -m  venv  venv
```

### 2. Ativar ambiente virtual

#### Windows
```
venv\Scripts\activate
```
#### Linux/macOS
```
source venv/bin/activate
```


### 3. Instalar dependências
```
pip  install  -r  requirements.txt
```

### 4. Executar a aplicação

```
python  -m  uvicorn  app.main:app  --reload
```

---

## 🚀 Endpoints da API

### ✅ Health Check
Verifica se a API está online.
```
GET /health
```

**Retorno:**
```json
{
	"success": true,
	"message": "API is running"
}
```

---  

### 📊 Model Information
Retorna as informações do modelo treinado.
```
GET /model/{rnn_type}
```

**Parâmetros:**
*  `rnn_type`: `lstm` ou `gru`

**Retorno:**
```json
{
	"success": true,
	"training_time": float,
	"training_datetime": "ISODateTime",
	"mean_absolute_error": float,
	"root_mean_squared_error": float,
	"train_config": {
		"rnn_type": string, // "lstm" ou "gru"
		"file_path": string,
		"column_data": string,
		"window_size": int,
		"multi_feature": bool,
		"epochs": int,
		"batch_size": int,
		"learning_rate": float,
		"dense_activation": string, // "relu"
		"rnn_units": [
			int
		],
		"dense_units": [
			int
		],
		"dropout_rate": float,
		"early_stopping_patience": int,
		"bidirecional": bool
	}
}
```

---

### 🎯 Targets
Lista as colunas disponíveis como target no CSV.
```
GET /targets?file_path={opcional}
```

**Parâmetros:**
*  `file_path`: caminho para o CSV (padrão: `train.csv`)

**Retorno:**
```json
{
	"success": true,
	"targets": [
		string
	]
}
```

---

### 🧠 Treinamento do Modelo
Treina um modelo com os parâmetros fornecidos.

```
POST /train
```

**Body:**
```json
{
	"rnn_type": string, // "lstm" ou "gru"
	"file_path": string,
	"column_data": string,
	"window_size": int,
	"multi_feature": bool,
	"epochs": int,
	"batch_size": int,
	"learning_rate": float,
	"dense_activation": string, // "relu"
	"rnn_units": [
		int
	],
	"dense_units": [
		int
	],
	"dropout_rate": float,
	"early_stopping_patience": int,
	"bidirecional": bool
}
```

**Retorno:**
```json
{
	"success": true,
	"training_time": float,
	"training_datetime": "ISODateTime",
	"mean_squared_error": float,
	"mean_absolute_error": float,
	"root_mean_squared_error": float,
	"mean_absolute_percentage_error": float,
	"r_2_score": float,
	"best_train_loss": float,
	"best_val_loss": float
}

```

---

### 🔮 Predição
Realiza a predição com um modelo previamente treinado.
```
POST /predict
```

**Body:**
```json
{
	"rnn_type": string, // "lstm" ou "gru"
	"file_path": string,
	"n_steps_ahead": int
}
```

**Retorno:**
```json
{
	"success": true,
	"prediction_time": float,
	"real_values": [
		[ 
			"ISODateTime", 
			float 
		]
	],
	"forecast_values": [
		[ 
			"ISODateTime", 
			float
		]
	]
}
```

---

### ⚙️ Pré-processamento
Executa o pipeline de pré-processamento do modelo. Internamente é usado pelo `/train`.
```
POST /preprocessing
```

**Body:**
```json
{
	"file_path": string,
	"column_data": string,
	"window_size": int,
	"multi_feature": bool
}
```

**Retorno:**
```json
{
	"success": true,
	"preprocessing_time": float
}

```
> Após a execução, arquivos como `x_train.csv`, `y_train.csv`, `x_test.csv`, `y_test.csv` são salvos no diretório `temp/csvs`.

---

## 💻 Tecnologias Utilizadas

*  **Python**
*  **FastAPI**
*  **TensorFlow/Keras**
*  **Scikit-learn**
*  **Pandas**
*  **Uvicorn**

---

## 🙋‍♀️ Contribuição
Caso encontre algum problema ou queira contribuir com melhorias, sinta-se à vontade para abrir uma **issue** ou enviar um **pull request**.

---

📈 **Chronos - Prevendo o futuro, uma série temporal por vez!** ⏳
