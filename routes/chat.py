from fastapi import APIRouter, HTTPException
from models.schemas import ChatStreamRequest
from llm.memory import get_memory
from llm.chain import get_chain
from utils.executor import run_in_executor
from fastapi import APIRouter, Request, HTTPException
from llm.memory import get_memory
from llm.chain import get_chain
from llm.stream_handler import SSEHandler
import asyncio
from sse_starlette.sse import EventSourceResponse

router = APIRouter()
@router.post("/chat/stream")
async def chat_stream(request: Request, body: ChatStreamRequest):
    session_id = body.session_id.strip()
    message = body.message.strip()

    if not session_id or not message:
        raise HTTPException(status_code=400, detail="session_id and message are required.")

    memory = get_memory(session_id)
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    handler = SSEHandler(queue=queue, loop=loop)
    chain = get_chain(memory, streaming=True, callbacks=[handler])

    async def event_generator():
        try:
            # Launch chain execution
            asyncio.create_task(chain.apredict(input=message))

            while True:
                if await request.is_disconnected():
                    break

                token = await queue.get()
                if token == "[END]":
                    break

                yield {"event": "message", "data": token}
        except Exception as e:
            yield {"event": "error", "data": str(e)}

    return EventSourceResponse(event_generator())