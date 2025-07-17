from langchain.text_splitter import RecursiveCharacterTextSplitter
import asyncio
from typing import List

async def smart_chunk_text(text: str, max_chunk_size: int = 1500, chunk_overlap: int = 200) -> List[str]:
    def _chunk():
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_text(text)
        filtered_chunks = [chunk.strip() for chunk in chunks if 50 <= len(chunk.strip()) <= max_chunk_size]
        return filtered_chunks
    return await asyncio.get_event_loop().run_in_executor(None, _chunk)
