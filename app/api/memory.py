from fastapi import APIRouter, HTTPException
from app.memory import get_memory
from langchain_community.chat_message_histories import PostgresChatMessageHistory
from app.config import DATABASE_URL
import asyncio
from app.logger import logger

router = APIRouter()

@router.post("/clear-memory")
async def clear_memory(session_id: str):
    try:
        history = PostgresChatMessageHistory(connection_string=DATABASE_URL, session_id=session_id)
        await asyncio.get_event_loop().run_in_executor(None, history.clear)
        return {"message": f"Memory cleared for session {session_id}"}
    except Exception as e:
        logger.error(f"Error clearing memory: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing memory: {str(e)}")

@router.get("/memory-info/{session_id}")
async def get_memory_info(session_id: str):
    try:
        memory = await get_memory(session_id)
        messages = memory.chat_memory.messages
        total_messages = len(messages)
        total_chars = sum(len(msg.content) for msg in messages)
        estimated_tokens = total_chars // 4
        return {
            "session_id": session_id,
            "total_messages": total_messages,
            "estimated_tokens": estimated_tokens,
            "max_token_limit": 200,
            "status": "OK" if estimated_tokens < 15000 else "WARNING"
        }
    except Exception as e:
        logger.error(f"Error getting memory info: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting memory info: {str(e)}")
