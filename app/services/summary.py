from app.config import OPENAI_API_KEY
from langchain_openai import ChatOpenAI
import logging

logger = logging.getLogger(__name__)

async def create_document_summary(chunks, session_id: str) -> str:
    try:
        sample_chunks = chunks[:2]
        sample_text = "\n".join(sample_chunks)
        summary_prompt = f"""
        Please provide a brief summary of this document content in 1-2 sentences:

        {sample_text[:1000]}
        """
        summary_llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            api_key=OPENAI_API_KEY,
            max_tokens=100
        )
        response = await summary_llm.ainvoke(summary_prompt)
        return response.content
    except Exception as e:
        logger.error(f"Error creating summary: {e}")
        return f"Document with {len(chunks)} chunks uploaded."
