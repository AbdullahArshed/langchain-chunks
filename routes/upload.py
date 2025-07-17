from fastapi import APIRouter, UploadFile, File, HTTPException
from io import BytesIO
from llm.memory import get_memory
from llm.chain import get_chain
from models.schemas import ChatResponse
from utils.executor import run_in_executor
from file_handlers.extractors import (
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_txt,
    extract_text_from_excel,
)

router = APIRouter()
@router.post("/upload", response_model=ChatResponse)
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

        chain = get_chain(memory)
        response = await run_in_executor(lambda: chain.predict(input=f"The user uploaded the following file content:\n{text}"))

        messages = memory.chat_memory.messages
        messages_list = [msg.dict() for msg in messages]
        return ChatResponse(response=response, messages=messages_list)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
