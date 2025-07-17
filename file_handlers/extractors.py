import fitz  # PyMuPDF
import pandas as pd
from io import BytesIO
from docx import Document

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