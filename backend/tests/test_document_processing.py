import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from pypdf import PdfWriter

from document_processing import (
    chunk_pages,
    compute_content_hash,
    delete_document,
    document_exists_by_hash,
    extract_pdf_pages,
    format_citation,
    get_chroma_collection,
    get_document_by_hash,
    retrieve_chunks,
    store_document_chunks,
    TextChunk,
)

import chromadb


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class DocumentProcessingTests(unittest.TestCase):
    def setUp(self):
        # Use a temp directory for ChromaDB so tests are isolated
        import tempfile
        import shutil

        self.tmpdir = tempfile.mkdtemp()
        self._client_patch = patch(
            "document_processing.get_chroma_client",
            lambda: chromadb.PersistentClient(path=self.tmpdir),
        )
        self._collection_patch = patch(
            "document_processing.get_chroma_collection",
            lambda: chromadb.PersistentClient(path=self.tmpdir)
            .get_or_create_collection(name="pdf_chunks"),
        )
        self._client_patch.start()
        self._collection_patch.start()

    def tearDown(self):
        self._client_patch.stop()
        self._collection_patch.stop()
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_extracts_text_with_one_based_page_numbers(self):
        pdf_path = (
            PROJECT_ROOT / "documents" / "Why-We-Use-Common-Paper-Standard-Agreements.pdf"
        )

        pages = extract_pdf_pages(pdf_path.read_bytes(), pdf_path.name)

        self.assertTrue(pages)
        self.assertEqual(pages[0]["page_number"], 1)
        self.assertEqual(pages[0]["source"], pdf_path.name)
        self.assertTrue(any(str(page["text"]).strip() for page in pages))

    def test_rejects_pdf_without_extractable_text(self):
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        pdf_buffer = BytesIO()
        writer.write(pdf_buffer)

        with self.assertRaisesRegex(ValueError, "No readable text"):
            extract_pdf_pages(pdf_buffer.getvalue(), "blank.pdf")

    def test_chunk_metadata_tracks_source_page_and_character_offset(self):
        pages = [
            {"source": "agreement.pdf", "page_number": 4, "text": "Clause one. " * 30},
            {"source": "agreement.pdf", "page_number": 5, "text": "Clause two. " * 30},
        ]

        chunks = chunk_pages(pages, chunk_size=80, chunk_overlap=10)

        self.assertGreater(len(chunks), 2)
        self.assertEqual({chunk.metadata["page_number"] for chunk in chunks}, {4, 5})
        self.assertEqual(
            {chunk.metadata["source"] for chunk in chunks},
            {"agreement.pdf"},
        )
        self.assertEqual(
            [chunk.metadata["chunk_id"] for chunk in chunks],
            list(range(1, len(chunks) + 1)),
        )
        self.assertTrue(all("start_index" in chunk.metadata for chunk in chunks))

    def test_compute_content_hash_is_deterministic(self):
        data = b"same content"
        self.assertEqual(compute_content_hash(data), compute_content_hash(data))
        self.assertNotEqual(compute_content_hash(data), compute_content_hash(b"other"))

    def test_citation_uses_source_basename_and_page(self):
        citation = format_citation(
            {"source": "uploads\\agreement.pdf", "page_number": 12}
        )

        self.assertEqual(citation, "agreement.pdf, page 12")

    def test_store_and_retrieve_chunks(self):
        chunks = [
            TextChunk(
                page_content="Termination requires thirty days written notice.",
                metadata={
                    "source": "agreement.pdf",
                    "page_number": 3,
                    "chunk_id": 1,
                    "start_index": 0,
                },
            ),
            TextChunk(
                page_content="Invoices are due each month.",
                metadata={
                    "source": "agreement.pdf",
                    "page_number": 8,
                    "chunk_id": 2,
                    "start_index": 0,
                },
            ),
        ]
        embeddings = [[0.9, 0.1], [0.1, 0.9]]

        store_document_chunks(
            document_id="test-doc-1",
            source_name="agreement.pdf",
            content_hash="abc123",
            chunks=chunks,
            embeddings=embeddings,
        )

        # Query for termination - should match chunk 1
        results = retrieve_chunks(
            document_ids="test-doc-1",
            query_embedding=[0.9, 0.1],
            k=1,
            minimum_score=0.0,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["citation"], "agreement.pdf, page 3")
        self.assertIn("Termination", results[0]["text"])

        # Document isolation: querying a different document_id returns nothing
        results_other = retrieve_chunks(
            document_ids="nonexistent-doc",
            query_embedding=[0.9, 0.1],
            k=1,
            minimum_score=0.0,
        )
        self.assertEqual(len(results_other), 0)

    def test_document_exists_and_get_by_hash(self):
        self.assertFalse(document_exists_by_hash("nonexistent-hash"))

        chunks = [
            TextChunk(
                page_content="Test content.",
                metadata={
                    "source": "test.pdf",
                    "page_number": 1,
                    "chunk_id": 1,
                    "start_index": 0,
                },
            )
        ]
        store_document_chunks(
            document_id="doc-1",
            source_name="test.pdf",
            content_hash="hash-123",
            chunks=chunks,
            embeddings=[[1.0, 0.0]],
        )

        self.assertTrue(document_exists_by_hash("hash-123"))
        doc = get_document_by_hash("hash-123")
        self.assertIsNotNone(doc)
        self.assertEqual(doc["document_id"], "doc-1")
        self.assertEqual(doc["source_name"], "test.pdf")

        # Delete
        deleted = delete_document("doc-1")
        self.assertEqual(deleted, 1)
        self.assertFalse(document_exists_by_hash("hash-123"))

    def test_chunking_rejects_overlap_not_less_than_chunk_size(self):
        with self.assertRaisesRegex(ValueError, "less than chunk size"):
            chunk_pages(
                [{"source": "agreement.pdf", "page_number": 1, "text": "text"}],
                chunk_size=50,
                chunk_overlap=50,
            )


if __name__ == "__main__":
    unittest.main()
