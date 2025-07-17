from fastapi import FastAPI
from routes import chat, upload, health

app = FastAPI(title="LangChain Memory Chat API")

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(upload.router)