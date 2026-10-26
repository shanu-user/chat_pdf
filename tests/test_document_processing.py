import unittest
from io import BytesIO
from pathlib import Path

from pypdf import PdfWriter

from document_processing import (
    chunk_pages,
    extract_pdf_pages,
    format_citation,
    retrieve_by_embeddings,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DocumentProcessingTests(unittest.TestCase):
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

    def test_embedding_retrieval_ranks_relevant_chunks_and_returns_citation(self):
        chunks = chunk_pages(
            [
                {
                    "source": "agreement.pdf",
                    "page_number": 3,
                    "text": "Termination requires thirty days written notice.",
                },
                {
                    "source": "agreement.pdf",
                    "page_number": 8,
                    "text": "Invoices are due each month.",
                },
            ],
            chunk_size=100,
            chunk_overlap=0,
        )

        results = retrieve_by_embeddings(
            chunks,
            [[0.9, 0.1], [0.1, 0.9], [1.0, 0.0]],
            k=1,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["citation"], "agreement.pdf, page 3")
        self.assertIn("Termination", results[0]["text"])

    def test_citation_uses_source_basename_and_page(self):
        citation = format_citation(
            {"source": "uploads\\agreement.pdf", "page_number": 12}
        )

        self.assertEqual(citation, "agreement.pdf, page 12")

    def test_embedding_retrieval_returns_no_sources_below_relevance_threshold(self):
        chunks = chunk_pages(
            [{"source": "agreement.pdf", "page_number": 1, "text": "A clause"}],
            chunk_size=100,
            chunk_overlap=0,
        )

        self.assertEqual(
            retrieve_by_embeddings(chunks, [[0.0, 1.0], [1.0, 0.0]], k=4),
            [],
        )

    def test_embedding_retrieval_rejects_missing_vectors(self):
        chunks = chunk_pages(
            [{"source": "agreement.pdf", "page_number": 1, "text": "A clause"}],
            chunk_size=100,
            chunk_overlap=0,
        )

        with self.assertRaisesRegex(ValueError, "one embedding per chunk"):
            retrieve_by_embeddings(chunks, [[1.0, 0.0]], k=4)

    def test_chunking_rejects_overlap_not_less_than_chunk_size(self):
        with self.assertRaisesRegex(ValueError, "less than chunk size"):
            chunk_pages(
                [{"source": "agreement.pdf", "page_number": 1, "text": "text"}],
                chunk_size=50,
                chunk_overlap=50,
            )


if __name__ == "__main__":
    unittest.main()
