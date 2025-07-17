import logging
from app.memory import get_memory
from app.services.summary import create_document_summary

logger = logging.getLogger(__name__)

async def process_chunks_batch(chunks, session_id: str, batch_size: int = 5) -> int:
    processed_count = 0
    try:
        memory = await get_memory(session_id)
        summary = await create_document_summary(chunks, session_id)
        memory.chat_memory.add_user_message(f"Document uploaded with {len(chunks)} chunks")
        memory.chat_memory.add_ai_message(f"Document processed: {summary}")
        processed_count = len(chunks)
    except Exception as e:
        logger.error(f"Error processing chunks: {e}")
        raise
    return processed_count
