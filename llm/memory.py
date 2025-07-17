from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import PostgresChatMessageHistory
from sqlalchemy.exc import SQLAlchemyError
from config import DATABASE_URL

def get_memory(session_id: str) -> ConversationBufferMemory:
    try:
        history = PostgresChatMessageHistory(connection_string=DATABASE_URL, session_id=session_id)
        return ConversationBufferMemory(return_messages=True, chat_memory=history)
    except SQLAlchemyError as e:
        raise RuntimeError(f"Database connection error: {e}")
