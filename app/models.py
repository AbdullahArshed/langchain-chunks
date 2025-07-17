from pydantic import BaseModel
from typing import List, Dict, Any

class ChatRequest(BaseModel):
    message: str
    session_id: str

class ChatResponse(BaseModel):
    response: str
    messages: List[Dict[str, Any]]
