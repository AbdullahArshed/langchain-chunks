from fastapi import APIRouter, HTTPException
from app.models import ChatRequest
from app.memory import get_memory
from langchain.chains import ConversationChain
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
from langchain_openai import ChatOpenAI
from app.config import OPENAI_API_KEY
from app.llm import llm
from app.logger import logger
import asyncio
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler

router = APIRouter()

@router.post("/stream-chat")
async def stream_chat(request: ChatRequest):
    user_input = request.message.strip()
    session_id = request.session_id.strip()

    if not user_input or not session_id:
        raise HTTPException(status_code=400, detail="Message and Session ID are required.")

    try:
        memory = await get_memory(session_id)
        callback = AsyncIteratorCallbackHandler()
        streaming_llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            api_key=OPENAI_API_KEY,
            streaming=True,
            callbacks=[callback]
        )
        chain = ConversationChain(llm=streaming_llm, memory=memory, verbose=False)

        async def token_stream():
            try:
                task = asyncio.create_task(chain.apredict(input=user_input))
                async for token in callback.aiter():
                    yield {"event": "token", "data": token}
                await task
                messages = memory.chat_memory.messages[-5:]
                final_messages = [msg.dict() for msg in messages]
                yield {"event": "complete", "data": str(final_messages)}
            except Exception as e:
                logger.error(f"Error in token stream: {e}")
                yield {"event": "error", "data": str(e)}

        return EventSourceResponse(token_stream())

    except Exception as e:
        logger.error(f"Error in stream chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))
