from pydantic import BaseModel
from typing import List, Dict, Any

class ChatResponse(BaseModel):
    response: str
    messages: List[Dict[str, Any]]
from pydantic import BaseModel

class ChatStreamRequest(BaseModel):
    session_id: str
    message: str
