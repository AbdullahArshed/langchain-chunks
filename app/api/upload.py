from fastapi import APIRouter, HTTPException, UploadFile, File
from app.models import ChatRequest, ChatResponse
from io import BytesIO
from app.services.file_extraction import extract_text_from_pdf, extract_text_from_docx, extract_text_from_txt, extract_text_from_excel
from app.services.chunking import smart_chunk_text
from app.services.memory_processing import process_chunks_batch
from app.memory import get_memory
from app.logger import logger

router = APIRouter()

@router.post("/upload", response_model=ChatResponse)
async def upload_file(session_id: str, file: UploadFile = File(...)):
    if not session_id.strip():
        raise HTTPException(status_code=400, detail="Session ID is required.")

    if file.size and file.size > 60 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size exceeds 60MB limit.")

    try:
        logger.info(f"Processing file: {file.filename} ({file.size} bytes)")

        contents = await file.read()
        file_buffer = BytesIO(contents)
        filename = file.filename.lower()

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

        chunks = await smart_chunk_text(text, max_chunk_size=1500, chunk_overlap=200)
        logger.info(f"Created {len(chunks)} chunks")

        if not chunks:
            raise HTTPException(status_code=400, detail="No valid content found in the document.")

        processed_count = await process_chunks_batch(chunks, session_id)
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
