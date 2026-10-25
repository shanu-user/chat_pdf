import unittest
from io import BytesIO
from pathlib import Path

from pypdf import PdfWriter

from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

from document_processing import (
    chunk_pages,
    extract_pdf_pages,
    format_citation,
    upload_fingerprint,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DeterministicEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, text):
        return [float(len(text)), 1.0]


class DocumentProcessingTests(unittest.TestCase):
    def test_upload_fingerprint_changes_when_filename_or_contents_change(self):
        original = upload_fingerprint("first.pdf", b"first content")

        self.assertNotEqual(original, upload_fingerprint("second.pdf", b"first content"))
        self.assertNotEqual(original, upload_fingerprint("first.pdf", b"second content"))

    def test_extracts_text_with_one_based_page_numbers(self):
        pdf_path = (
            PROJECT_ROOT
            / "documents"
            / "Why-We-Use-Common-Paper-Standard-Agreements.pdf"
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

    def test_chunk_metadata_keeps_source_page_and_chunk_id(self):
        pages = [
            {
                "source": "agreement.pdf",
                "page_number": 4,
                "text": "A clause " * 30,
            },
            {
                "source": "agreement.pdf",
                "page_number": 5,
                "text": "Another clause " * 30,
            },
        ]

        chunks = chunk_pages(pages, chunk_size=80, chunk_overlap=10)

        self.assertGreater(len(chunks), 2)
        self.assertEqual(
            {chunk.metadata["page_number"] for chunk in chunks},
            {4, 5},
        )
        self.assertEqual(
            {chunk.metadata["source"] for chunk in chunks},
            {"agreement.pdf"},
        )
        self.assertEqual(
            [chunk.metadata["chunk_id"] for chunk in chunks],
            list(range(1, len(chunks) + 1)),
        )
        self.assertTrue(all("start_index" in chunk.metadata for chunk in chunks))

    def test_citation_uses_source_basename_and_page(self):
        citation = format_citation(
            {"source": "uploads\\agreement.pdf", "page_number": 12}
        )

        self.assertEqual(citation, "agreement.pdf, page 12")

    def test_faiss_retrieval_preserves_page_metadata(self):
        chunks = chunk_pages(
            [{"source": "agreement.pdf", "page_number": 8, "text": "Termination clause."}],
            chunk_size=100,
            chunk_overlap=0,
        )
        vector_store = FAISS.from_documents(chunks, DeterministicEmbeddings())

        retrieved = vector_store.similarity_search("termination", k=1)

        self.assertEqual(retrieved[0].metadata["source"], "agreement.pdf")
        self.assertEqual(retrieved[0].metadata["page_number"], 8)
        self.assertEqual(format_citation(retrieved[0].metadata), "agreement.pdf, page 8")

    def test_chunking_rejects_overlap_not_less_than_chunk_size(self):
        with self.assertRaisesRegex(ValueError, "less than chunk size"):
            chunk_pages(
                [{"source": "agreement.pdf", "page_number": 1, "text": "text"}],
                chunk_size=50,
                chunk_overlap=50,
            )


if __name__ == "__main__":
    unittest.main()
