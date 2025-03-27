from fastapi import FastAPI
from app.api.routes import router

app = FastAPI()

# Inclui as rotas da API
app.include_router(router)
