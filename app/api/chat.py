from fastapi import APIRouter, HTTPException
from app.models import ChatRequest, ChatResponse
from app.memory import get_memory
from app.llm import llm
from langchain.chains import ConversationChain
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    user_input = request.message.strip()
    session_id = request.session_id.strip()

    if not user_input:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID is required.")

    try:
        memory = await get_memory(session_id)
        chain = ConversationChain(llm=llm, memory=memory, verbose=False)
        response = await chain.apredict(input=user_input)
        messages = memory.chat_memory.messages[-5:]
        messages_list = [msg.dict() for msg in messages]
        return ChatResponse(response=response, messages=messages_list)
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))
