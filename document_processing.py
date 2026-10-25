import hashlib
from io import BytesIO
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


def upload_fingerprint(filename: str, pdf_bytes: bytes) -> str:
    return hashlib.sha256(filename.encode("utf-8") + pdf_bytes).hexdigest()


def extract_pdf_pages(pdf_bytes: bytes, source_name: str) -> list[dict[str, str | int]]:
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = [
        {
            "source": source_name,
            "page_number": page_number,
            "text": page.extract_text() or "",
        }
        for page_number, page in enumerate(reader.pages, start=1)
    ]

    if not any(page["text"].strip() for page in pages):
        raise ValueError("No readable text found in this PDF.")

    return pages


def chunk_pages(
    pages: list[dict[str, str | int]],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be at least zero and less than chunk size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    chunks = []

    for page in pages:
        page_text = str(page["text"])
        if not page_text.strip():
            continue

        page_number = int(page["page_number"])
        source = str(page["source"])
        page_chunks = splitter.create_documents(
            [page_text],
            metadatas=[{"source": source, "page_number": page_number}],
        )
        for chunk in page_chunks:
            chunk.metadata["chunk_id"] = len(chunks) + 1
            chunks.append(chunk)

    return chunks


def format_citation(metadata: dict[str, object]) -> str:
    source = Path(str(metadata.get("source", "Unknown source"))).name
    page_number = metadata.get("page_number")
    if isinstance(page_number, int):
        return f"{source}, page {page_number}"
    return f"{source}, page unavailable"
