import fitz  # PyMuPDF
from io import BytesIO
from docx import Document
import pandas as pd
import asyncio

async def extract_text_from_pdf(file: BytesIO) -> str:
    def _extract():
        doc = fitz.open(stream=file, filetype="pdf")
        text_parts = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            if text.strip():
                text_parts.append(text)
            page = None
        doc.close()
        return "\n".join(text_parts)
    return await asyncio.get_event_loop().run_in_executor(None, _extract)

async def extract_text_from_docx(file: BytesIO) -> str:
    def _extract():
        document = Document(file)
        return "\n".join(paragraph.text for paragraph in document.paragraphs)
    return await asyncio.get_event_loop().run_in_executor(None, _extract)

async def extract_text_from_txt(file: BytesIO) -> str:
    def _extract():
        return file.read().decode("utf-8")
    return await asyncio.get_event_loop().run_in_executor(None, _extract)

async def extract_text_from_excel(file: BytesIO) -> str:
    def _extract():
        df = pd.read_excel(file, sheet_name=None)
        return "\n".join([f"Sheet: {sheet}\n{data.to_string(index=False)}" for sheet, data in df.items()])
    return await asyncio.get_event_loop().run_in_executor(None, _extract)
