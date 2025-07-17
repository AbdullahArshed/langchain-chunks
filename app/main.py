from fastapi import FastAPI
from app.api import  stream_chat, upload,  memory
from app.logger import logger

app = FastAPI(title="LangChain Memory Chat API")

# Include routers

app.include_router(stream_chat.router)
app.include_router(upload.router)
app.include_router(memory.router)

@app.get("/")
def root():
    return {"message": "LangChain Memory Chat API is running."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
