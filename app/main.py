from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router

app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Você pode trocar "*" pelo domínio do seu frontend
    allow_credentials=True,
    allow_methods=["*"],  # Isso permite POST, GET, OPTIONS, etc.
    allow_headers=["*"],
)

# Inclui as rotas da API
app.include_router(router)
