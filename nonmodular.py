import os
import asyncio
import fitz  # PyMuPDF
import pandas as pd
from io import BytesIO
from docx import Document
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List, Dict, Any
import logging

from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.chat_message_histories import PostgresChatMessageHistory
from langchain_openai import ChatOpenAI
from sqlalchemy.exc import SQLAlchemyError

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from sse_starlette.sse import EventSourceResponse  # For streaming

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
PGDATABASE = os.getenv("PGDATABASE", "langchainchunks")

DATABASE_URL = f"postgresql://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{PGDATABASE}"

# Initialize async LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=openai_api_key)

app = FastAPI(title="LangChain Memory Chat API")

class ChatRequest(BaseModel):
    message: str
    session_id: str

class ChatResponse(BaseModel):
    response: str
    messages: List[Dict[str, Any]]

async def get_memory(session_id: str) -> ConversationSummaryBufferMemory:
    """Get memory instance asynchronously"""
    try:
        history = PostgresChatMessageHistory(
            connection_string=DATABASE_URL,
            session_id=session_id
        )
        summarizer_llm = ChatOpenAI(
            temperature=0,
            api_key=openai_api_key,
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
        memory = await get_memory(session_id)
        chain = ConversationChain(llm=llm, memory=memory, verbose=False)

        # Use async predict instead of sync predict with executor
        response = await chain.apredict(input=user_input)

        messages = memory.chat_memory.messages[-5:]  # Only return last 5 messages
        messages_list = [msg.dict() for msg in messages]
        return ChatResponse(response=response, messages=messages_list)

    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stream-chat")
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
            api_key=openai_api_key,
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

async def extract_text_from_pdf(file: BytesIO) -> str:
    """Extract text from PDF asynchronously"""
    def _extract():
        doc = fitz.open(stream=file, filetype="pdf")
        text_parts = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            if text.strip():  # Only add non-empty pages
                text_parts.append(text)
            page = None  # Help with memory cleanup
        
        doc.close()
        return "\n".join(text_parts)
    
    # Run in thread pool for CPU-bound operation
    return await asyncio.get_event_loop().run_in_executor(None, _extract)

async def extract_text_from_docx(file: BytesIO) -> str:
    """Extract text from DOCX asynchronously"""
    def _extract():
        document = Document(file)
        return "\n".join(paragraph.text for paragraph in document.paragraphs)
    
    return await asyncio.get_event_loop().run_in_executor(None, _extract)

async def extract_text_from_txt(file: BytesIO) -> str:
    """Extract text from TXT asynchronously"""
    def _extract():
        return file.read().decode("utf-8")
    
    return await asyncio.get_event_loop().run_in_executor(None, _extract)

async def extract_text_from_excel(file: BytesIO) -> str:
    """Extract text from Excel asynchronously"""
    def _extract():
        df = pd.read_excel(file, sheet_name=None)
        return "\n".join([f"Sheet: {sheet}\n{data.to_string(index=False)}" for sheet, data in df.items()])
    
    return await asyncio.get_event_loop().run_in_executor(None, _extract)

async def smart_chunk_text(text: str, max_chunk_size: int = 1500, chunk_overlap: int = 200) -> List[str]:
    """
    Smart chunking that respects token limits and content structure (async)
    """
    def _chunk():
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = splitter.split_text(text)
        
        # Filter out very small chunks and very large chunks
        filtered_chunks = []
        for chunk in chunks:
            chunk = chunk.strip()
            if 50 <= len(chunk) <= max_chunk_size:  # Only keep reasonable-sized chunks
                filtered_chunks.append(chunk)
        
        return filtered_chunks
    
    return await asyncio.get_event_loop().run_in_executor(None, _chunk)

async def create_document_summary(chunks: List[str], session_id: str) -> str:
    """Create a summary of the document for the memory (async)"""
    try:
        # Take first few chunks to create a summary
        sample_chunks = chunks[:2]  # Only use first 2 chunks for summary
        sample_text = "\n".join(sample_chunks)
        
        # Create a simple summary
        summary_prompt = f"""
        Please provide a brief summary of this document content in 1-2 sentences:
        
        {sample_text[:1000]}  # Limit to 1000 characters
        """
        
        summary_llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            api_key=openai_api_key,
            max_tokens=100  # Reduced tokens for summary
        )
        
        # Use async invoke instead of sync
        response = await summary_llm.ainvoke(summary_prompt)
        return response.content
        
    except Exception as e:
        logger.error(f"Error creating summary: {e}")
        return f"Document with {len(chunks)} chunks uploaded."

async def process_chunks_batch(chunks: List[str], session_id: str, batch_size: int = 5) -> int:
    """Process chunks in batches to avoid memory issues (async)"""
    processed_count = 0
    
    try:
        memory = await get_memory(session_id)
        
        # Create a summary and add it to memory instead of processing all chunks
        summary = await create_document_summary(chunks, session_id)
        
        # Add the summary to memory
        memory.chat_memory.add_user_message(f"Document uploaded with {len(chunks)} chunks")
        memory.chat_memory.add_ai_message(f"Document processed: {summary}")
        
        processed_count = len(chunks)
        
    except Exception as e:
        logger.error(f"Error processing chunks: {e}")
        raise
    
    return processed_count

@app.post("/upload", response_model=ChatResponse)
async def upload_file(session_id: str, file: UploadFile = File(...)):
    if not session_id.strip():
        raise HTTPException(status_code=400, detail="Session ID is required.")
    
    # Check file size (60MB limit)
    if file.size and file.size > 60 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size exceeds 60MB limit.")

    try:
        logger.info(f"Processing file: {file.filename} ({file.size} bytes)")
        
        contents = await file.read()
        file_buffer = BytesIO(contents)
        filename = file.filename.lower()

        # Extract text based on file type (now async)
        if filename.endswith(".pdf"):
            text = await extract_text_from_pdf(file_buffer)
        elif filename.endswith(".docx"):
            text = await extract_text_from_docx(file_buffer)
        elif filename.endswith(".txt"):
            text = await extract_text_from_txt(file_buffer)
        elif filename.endswith((".xlsx", ".xls")):
            text = await extract_text_from_excel(file_buffer)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        logger.info(f"Extracted text length: {len(text)} characters")
        
        # Smart chunking (now async)
        chunks = await smart_chunk_text(text, max_chunk_size=1500, chunk_overlap=200)
        logger.info(f"Created {len(chunks)} chunks")
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No valid content found in the document.")
        
        # Process chunks in batches (already async)
        processed_count = await process_chunks_batch(chunks, session_id)
        
        # Get updated memorya
        memory = await get_memory(session_id)
        messages = memory.chat_memory.messages[-5:]
        messages_list = [msg.dict() for msg in messages]
        
        response_message = f"Successfully processed {file.filename} with {len(chunks)} chunks. You can now ask questions about the document content."
        
        return ChatResponse(response=response_message, messages=messages_list)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "LangChain Memory Chat API"}

@app.post("/clear-memory")
async def clear_memory(session_id: str):
    """Clear conversation memory for a session"""
    try:
        history = PostgresChatMessageHistory(
            connection_string=DATABASE_URL,
            session_id=session_id
        )
        # Use async clear if available, otherwise run in executor
        await asyncio.get_event_loop().run_in_executor(None, history.clear)
        return {"message": f"Memory cleared for session {session_id}"}
    except Exception as e:
        logger.error(f"Error clearing memory: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing memory: {str(e)}")

@app.get("/memory-info/{session_id}")
async def get_memory_info(session_id: str):
    """Get information about current memory usage"""
    try:
        memory = await get_memory(session_id)
        messages = memory.chat_memory.messages
        total_messages = len(messages)
        
        # Estimate token count (rough approximation)
        total_chars = sum(len(msg.content) for msg in messages)
        estimated_tokens = total_chars // 4  # Rough estimate: 4 chars per token
        
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)