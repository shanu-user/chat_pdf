import math
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader


@dataclass
class TextChunk:
    page_content: str
    metadata: dict[str, str | int]


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
) -> list[TextChunk]:
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be at least zero and less than chunk size.")

    chunks = []
    for page in pages:
        text = str(page["text"])
        if not text.strip():
            continue

        source = str(page["source"])
        page_number = int(page["page_number"])
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                min_boundary = start + chunk_size // 2
                boundary = max(
                    text.rfind("\n", min_boundary, end),
                    text.rfind(" ", min_boundary, end),
                )
                if boundary > start:
                    end = boundary

            raw_chunk = text[start:end]
            leading_whitespace = len(raw_chunk) - len(raw_chunk.lstrip())
            content = raw_chunk.strip()
            if content:
                chunk_id = len(chunks) + 1
                chunks.append(
                    TextChunk(
                        page_content=content,
                        metadata={
                            "source": source,
                            "page_number": page_number,
                            "chunk_id": chunk_id,
                            "start_index": start + leading_whitespace,
                        },
                    )
                )

            if end >= len(text):
                break
            start = max(start + 1, end - chunk_overlap)

    return chunks


def retrieve_by_embeddings(
    chunks: list[TextChunk],
    embeddings: list[list[float]],
    k: int,
    minimum_score: float = 0.2,
) -> list[dict[str, str | int]]:
    if k <= 0:
        raise ValueError("Result count must be positive.")
    if len(embeddings) != len(chunks) + 1:
        raise ValueError("Expected one embedding per chunk and one for the query.")
    if not chunks:
        return []

    query_embedding = embeddings[-1]
    query_norm = math.sqrt(sum(value * value for value in query_embedding))
    if query_norm == 0:
        return []

    scored_chunks = []
    for index, chunk in enumerate(chunks):
        embedding = embeddings[index]
        if len(embedding) != len(query_embedding):
            raise ValueError("Chunk and query embeddings have different dimensions.")
        embedding_norm = math.sqrt(sum(value * value for value in embedding))
        if embedding_norm == 0:
            continue

        score = sum(
            value * query_value
            for value, query_value in zip(embedding, query_embedding)
        ) / (embedding_norm * query_norm)
        if score >= minimum_score:
            scored_chunks.append((score, index, chunk))

    scored_chunks.sort(key=lambda item: (-item[0], item[1]))
    return [
        {
            "citation": format_citation(chunk.metadata),
            "text": chunk.page_content,
            "chunk_id": int(chunk.metadata["chunk_id"]),
        }
        for _, _, chunk in scored_chunks[:k]
    ]


def format_citation(metadata: dict[str, str | int]) -> str:
    source_raw = str(metadata.get("source", "Unknown source"))
    # Normalize both Windows and POSIX separators for cross-platform basename extraction
    source_normalized = source_raw.replace("\\", "/")
    source = Path(source_normalized).name
    page_number = metadata.get("page_number")
    if isinstance(page_number, int):
        return f"{source}, page {page_number}"
    return source
