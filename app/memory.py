from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.chat_message_histories import PostgresChatMessageHistory
from langchain_openai import ChatOpenAI
from sqlalchemy.exc import SQLAlchemyError
from app.config import DATABASE_URL, OPENAI_API_KEY

async def get_memory(session_id: str) -> ConversationSummaryBufferMemory:
    try:
        history = PostgresChatMessageHistory(
            connection_string=DATABASE_URL,
            session_id=session_id
        )
        summarizer_llm = ChatOpenAI(
            temperature=0,
            api_key=OPENAI_API_KEY,
            model="gpt-3.5-turbo"
        )
        return ConversationSummaryBufferMemory(
            llm=summarizer_llm,
            chat_memory=history,
            return_messages=True,
            max_token_limit=200
        )
    except SQLAlchemyError as e:
        raise RuntimeError(f"Database connection error: {e}")
