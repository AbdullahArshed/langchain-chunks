
import os
import asyncio
import fitz  # PyMuPDF
import pandas as pd
from io import BytesIO
from docx import Document
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import PostgresChatMessageHistory
from langchain_openai import ChatOpenAI
from sqlalchemy.exc import SQLAlchemyError

from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from sse_starlette.sse import EventSourceResponse  # For streaming

# Load environment variables
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise EnvironmentError("OPENAI_API_KEY not found")

# PostgreSQL settings
PGUSER = os.getenv("PGUSER", "postgres")
PGPASSWORD = os.getenv("PGPASSWORD", "")
PGHOST = os.getenv("PGHOST", "127.0.0.1")
PGPORT = os.getenv("PGPORT", "5432")
PGDATABASE = os.getenv("PGDATABASE", "streamlangchain")

DATABASE_URL = f"postgresql://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{PGDATABASE}"
print("DATABASE_URL:", DATABASE_URL)

executor = ThreadPoolExecutor()

# Default non-streaming LLM instance
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=openai_api_key)

app = FastAPI(title="LangChain Memory Chat API")

class ChatRequest(BaseModel):
    message: str
    session_id: str

class ChatResponse(BaseModel):
    response: str
    messages: List[Dict[str, Any]]

def get_memory(session_id: str) -> ConversationBufferMemory:
    try:
        history = PostgresChatMessageHistory(connection_string=DATABASE_URL, session_id=session_id)
        return ConversationBufferMemory(return_messages=True, chat_memory=history)
    except SQLAlchemyError as e:
        raise RuntimeError(f"Database connection error: {e}")

@app.get("/")
def root():
    return {"message": "LangChain Memory Chat API is running."}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    user_input = request.message.strip()
    session_id = request.session_id.strip()

    if not user_input:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID is required.")

    try:
        memory = get_memory(session_id)
        chain = ConversationChain(llm=llm, memory=memory, verbose=True)

        response = await asyncio.get_event_loop().run_in_executor(
            executor, lambda: chain.predict(input=user_input)
        )

        messages = memory.chat_memory.messages
        messages_list = [msg.dict() for msg in messages]
        return ChatResponse(response=response, messages=messages_list)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Streaming Chat Endpoint ----------

@app.post("/stream-chat")
async def stream_chat(request: ChatRequest):
    user_input = request.message.strip()
    session_id = request.session_id.strip()

    if not user_input or not session_id:
        raise HTTPException(status_code=400, detail="Message and Session ID are required.")

    try:
        memory = get_memory(session_id)
        callback = AsyncIteratorCallbackHandler()

        streaming_llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            api_key=openai_api_key,
            streaming=True,
            callbacks=[callback]
        )

        chain = ConversationChain(llm=streaming_llm, memory=memory, verbose=True)

        async def token_stream():
            task = asyncio.create_task(chain.apredict(input=user_input))
            async for token in callback.aiter():
                yield {"event": "token", "data": token}
            await task
            messages = memory.chat_memory.messages
            final_messages = [msg.dict() for msg in messages]
            yield {"event": "complete", "data": str(final_messages)}

        return EventSourceResponse(token_stream())

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- File Processing Utilities ----------

def extract_text_from_pdf(file: BytesIO) -> str:
    doc = fitz.open(stream=file, filetype="pdf")
    return "\n".join(page.get_text() for page in doc)

def extract_text_from_docx(file: BytesIO) -> str:
    document = Document(file)
    return "\n".join(paragraph.text for paragraph in document.paragraphs)

def extract_text_from_txt(file: BytesIO) -> str:
    return file.read().decode("utf-8")

def extract_text_from_excel(file: BytesIO) -> str:
    df = pd.read_excel(file, sheet_name=None)
    return "\n".join([f"Sheet: {sheet}\n{data.to_string(index=False)}" for sheet, data in df.items()])


# ---------- File Upload Endpoint ----------

@app.post("/upload", response_model=ChatResponse)
async def upload_file(session_id: str, file: UploadFile = File(...)):
    memory = get_memory(session_id)

    try:
        contents = await file.read()
        file_buffer = BytesIO(contents)
        filename = file.filename.lower()

        if filename.endswith(".pdf"):
            text = extract_text_from_pdf(file_buffer)
        elif filename.endswith(".docx"):
            text = extract_text_from_docx(file_buffer)
        elif filename.endswith(".txt"):
            text = extract_text_from_txt(file_buffer)
        elif filename.endswith((".xlsx", ".xls")):
            text = extract_text_from_excel(file_buffer)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        chain = ConversationChain(llm=llm, memory=memory, verbose=True)

        response = await asyncio.get_event_loop().run_in_executor(
            executor, lambda: chain.predict(input=f"The user uploaded the following file content:\n{text}")
        )

        messages = memory.chat_memory.messages
        messages_list = [msg.dict() for msg in messages]
        return ChatResponse(response=response, messages=messages_list)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
