from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def root():
    return {"message": "LangChain Memory Chat API is running."}
