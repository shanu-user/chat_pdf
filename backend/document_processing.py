import hashlib
import logging
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Optional
from uuid import uuid4

import chromadb
from pypdf import PdfReader


logger = logging.getLogger(__name__)

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "pdf_chunks")
MAX_CHUNK_SIZE_CHARS = 1000
MAX_CHUNK_OVERLAP_CHARS = 150


@dataclass
class TextChunk:
    page_content: str
    metadata: dict[str, str | int]


@dataclass
class DocumentRecord:
    document_id: str
    source_name: str
    content_hash: str
    num_chunks: int
    num_pages: int


def compute_content_hash(pdf_bytes: bytes) -> str:
    return hashlib.sha256(pdf_bytes).hexdigest()


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
    chunk_size: int = MAX_CHUNK_SIZE_CHARS,
    chunk_overlap: int = MAX_CHUNK_OVERLAP_CHARS,
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


@lru_cache(maxsize=1)
def get_chroma_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


@lru_cache(maxsize=1)
def get_chroma_collection():
    client = get_chroma_client()
    return client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)


def document_exists_by_hash(content_hash: str) -> bool:
    try:
        collection = get_chroma_collection()
        results = collection.get(
            where={"content_hash": content_hash},
            limit=1,
        )
        return bool(results and results.get("ids") and len(results["ids"]) > 0)
    except Exception as error:
        logger.warning("Failed to check document existence: %s", error)
        return False


def get_document_by_hash(content_hash: str):
    try:
        collection = get_chroma_collection()
        results = collection.get(where={"content_hash": content_hash}, limit=1)
        if not results or not results.get("metadatas"):
            return None
        metadata = results["metadatas"][0]
        doc_id = str(metadata.get("document_id", ""))
        return {
            "document_id": doc_id,
            "source_name": str(metadata.get("source", "Unknown")),
            "content_hash": str(metadata.get("content_hash", "")),
            "num_chunks": len(results.get("ids", [])),
            "num_pages": len(
                set(
                    int(m.get("page_number", 0))
                    for m in results.get("metadatas", [])
                    if m.get("page_number") is not None
                )
            ),
        }
    except Exception as error:
        logger.warning("Failed to get document by hash: %s", error)
        return None


def store_document_chunks(
    document_id: str,
    source_name: str,
    content_hash: str,
    chunks: list[TextChunk],
    embeddings: list[list[float]],
) -> None:
    if len(chunks) != len(embeddings):
        raise ValueError("Chunk count and embedding count must match.")

    collection = get_chroma_collection()
    ids = [f"{document_id}:{chunk.metadata['chunk_id']}" for chunk in chunks]
    documents = [chunk.page_content for chunk in chunks]
    metadatas = [
        {
            "document_id": document_id,
            "source": chunk.metadata["source"],
            "page_number": int(chunk.metadata["page_number"]),
            "chunk_id": int(chunk.metadata["chunk_id"]),
            "start_index": int(chunk.metadata["start_index"]),
            "content_hash": content_hash,
        }
        for chunk in chunks
    ]
    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)


def retrieve_chunks(
    document_ids: str,
    query_embedding: list[float],
    k: int = 5,
    minimum_score: float = 0.2,
) -> list[dict[str, str | int]]:
    if k <= 0:
        raise ValueError("Result count must be positive.")

    # Support comma-separated document IDs for multi-document queries
    id_list = [d.strip() for d in document_ids.split(",") if d.strip()]
    if not id_list:
        return []

    collection = get_chroma_collection()
    all_scored = []

    for doc_id in id_list:
        results = collection.query(
            query_embeddings=[query_embedding],
            where={"document_id": doc_id},
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        if not results or not results.get("ids") or len(results["ids"][0]) == 0:
            continue

        query_norm = math.sqrt(sum(v * v for v in query_embedding))
        if query_norm == 0:
            continue

        ids = results["ids"][0]
        documents = results["documents"][0] or []
        metadatas = results["metadatas"][0] or []
        distances = results["distances"][0] or []

        for doc_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
            if metadata is None:
                continue
            similarity = 1.0 - float(distance)
            if similarity < minimum_score:
                continue

            all_scored.append(
                {
                    "citation": format_citation(metadata),
                    "text": text,
                    "chunk_id": int(metadata.get("chunk_id", 0)),
                    "similarity": similarity,
                    "document_id": metadata.get("document_id", doc_id),
                }
            )

    all_scored.sort(key=lambda item: (-item["similarity"], item["chunk_id"]))
    return all_scored[:k]


def format_citation(metadata: dict[str, str | int]) -> str:
    source_raw = str(metadata.get("source", "Unknown source"))
    # Normalize both Windows and POSIX separators for cross-platform basename extraction
    source_normalized = source_raw.replace("\\", "/")
    source = Path(source_normalized).name
    page_number = metadata.get("page_number")
    if isinstance(page_number, int):
        return f"{source}, page {page_number}"
    return source


def delete_document(document_id: str) -> int:
    """Delete all chunks for a document. Returns number of deleted chunks."""
    collection = get_chroma_collection()
    results = collection.get(where={"document_id": document_id})
    if not results or not results.get("ids"):
        return 0
    collection.delete(ids=results["ids"])
    return len(results["ids"])


def get_document_info(document_id: str) -> Optional[DocumentRecord]:
    collection = get_chroma_collection()
    results = collection.get(where={"document_id": document_id}, limit=1)
    if not results or not results.get("metadatas"):
        return None
    metadata = results["metadatas"][0]
    return DocumentRecord(
        document_id=document_id,
        source_name=str(metadata.get("source", "Unknown")),
        content_hash=str(metadata.get("content_hash", "")),
        num_chunks=len(results.get("ids", [])),
        num_pages=len(
            set(
                int(m.get("page_number", 0))
                for m in results.get("metadatas", [])
                if m.get("page_number") is not None
            )
        ),
    )
