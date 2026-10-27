import os
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch

import sys

# Ensure backend is on the path when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app as app_module

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_PDF = (
    PROJECT_ROOT / "documents" / "Why-We-Use-Common-Paper-Standard-Agreements.pdf"
)
SAMPLE_BYTES = SAMPLE_PDF.read_bytes()

# Create a valid PDF with different content for isolation testing:
# Append a comment after %%EOF (PDF readers ignore trailing content)
DIFFERENT_BYTES = SAMPLE_BYTES + b"\n% Different isolation document marker\n"


class AppTests(unittest.TestCase):
    def setUp(self):
        self.client = app_module.app.test_client()
        self.env_patch = patch.dict(
            os.environ,
            {"OLLAMA_API_KEY": "test-api-key", "APP_ACCESS_CODE": "test-access"},
            clear=False,
        )
        self.env_patch.start()
        self._document_ids = []

    def tearDown(self):
        self.env_patch.stop()
        # Clean up any test documents from ChromaDB
        for doc_id in self._document_ids:
            try:
                from document_processing import delete_document

                delete_document(doc_id)
            except Exception:
                pass

    def _upload_sample_pdf(self, filename=None, content=None):
        """Upload a PDF and return the document_id."""
        if filename is None:
            filename = SAMPLE_PDF.name
        if content is None:
            content = SAMPLE_BYTES
        data = {
            "sources": [(BytesIO(content), filename)],
        }
        headers = {"X-App-Access-Code": "test-access"}
        response = self.client.post("/api/upload", data=data, headers=headers)
        self.assertEqual(response.status_code, 200, response.data)
        documents = response.json["documents"]
        self.assertEqual(len(documents), 1)
        doc_info = documents[0]
        self._document_ids.append(doc_info["document_id"])
        return doc_info["document_id"], doc_info["status"]

    def post_ask(self, document_id, question="What are the standard agreements?", access_code="test-access"):
        data = {
            "question": question,
            "document_id": document_id,
        }
        headers = {}
        if access_code is not None:
            headers["X-App-Access-Code"] = access_code
        return self.client.post("/api/ask", data=data, headers=headers)

    def test_health_and_frontend_routes(self):
        health = self.client.get("/api/health")

        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json["status"], "ok")
        self.assertEqual(health.json["chroma"], "ok")
        health.close()

    def test_upload_and_ask_returns_answer_and_sources(self):
        document_id, status = self._upload_sample_pdf()
        self.assertEqual(status, "indexed")

        # Mock retrieve_chunks and generate_answer
        fake_retrieved = [
            {
                "citation": "Why-We-Use-Common-Paper-Standard-Agreements.pdf, page 1",
                "text": "The document describes standard agreements.",
                "chunk_id": 1,
                "similarity": 0.95,
                "document_id": document_id,
            }
        ]
        with (
            patch.object(app_module, "retrieve_chunks", return_value=fake_retrieved),
            patch.object(
                app_module,
                "generate_answer",
                return_value="The document describes standard agreements.",
            ),
        ):
            response = self.post_ask(document_id)

        self.assertEqual(response.status_code, 200, response.data)
        self.assertEqual(
            response.json["answer"],
            "The document describes standard agreements.",
        )
        self.assertTrue(response.json["sources"])

    def test_graph_routes_to_no_match_without_calling_chat_model(self):
        document_id, _ = self._upload_sample_pdf()

        # Mock retrieve_chunks to return no results
        with patch.object(app_module, "retrieve_chunks", return_value=[]):
            response = self.post_ask(document_id, question="A completely unrelated question")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["sources"], [])
        self.assertIn("couldn't find relevant passages", response.json["answer"])

    def test_ollama_chat_uses_server_side_bearer_auth(self):
        response = Mock()
        response.json.return_value = {"message": {"content": "A grounded answer."}}
        with patch.object(app_module.httpx, "post", return_value=response) as post:
            answer = app_module.generate_answer(
                "What is the notice period?",
                [{"citation": "agreement.pdf, page 2", "text": "Thirty days notice."}],
            )

        self.assertEqual(answer, "A grounded answer.")
        request = post.call_args
        self.assertEqual(request.args[0], "https://ollama.com/api/chat")
        self.assertEqual(
            request.kwargs["headers"]["Authorization"],
            "Bearer test-api-key",
        )
        self.assertEqual(request.kwargs["json"]["model"], app_module.CHAT_MODEL)
        self.assertIn(
            "agreement.pdf, page 2",
            request.kwargs["json"]["messages"][1]["content"],
        )

    def test_document_isolation(self):
        """Two documents with different content remain isolated."""
        doc_id_1, status_1 = self._upload_sample_pdf(filename="doc-a.pdf", content=SAMPLE_BYTES)
        doc_id_2, status_2 = self._upload_sample_pdf(filename="doc-b.pdf", content=DIFFERENT_BYTES)

        self.assertEqual(status_1, "indexed")
        self.assertEqual(status_2, "indexed")
        self.assertNotEqual(doc_id_1, doc_id_2)

        # Mock retrieve_chunks and generate_answer to verify document_id filtering
        def fake_retrieve(document_id, query_embedding, k=5, minimum_score=0.2):
            return [
                {
                    "citation": f"{document_id}, page 1",
                    "text": f"Content from {document_id}",
                    "chunk_id": 1,
                    "similarity": 0.9,
                    "document_id": document_id,
                }
            ]

        with (
            patch.object(app_module, "retrieve_chunks", side_effect=fake_retrieve),
            patch.object(
                app_module,
                "generate_answer",
                return_value="Grounded answer.",
            ),
        ):
            # Query doc-1
            response = self.post_ask(doc_id_1, "What is in this document?")
            self.assertEqual(response.status_code, 200)
            for source in response.json["sources"]:
                self.assertEqual(source["document_id"], doc_id_1)

            # Query doc-2
            response = self.post_ask(doc_id_2, "What is in this document?")
            self.assertEqual(response.status_code, 200)
            for source in response.json["sources"]:
                self.assertEqual(source["document_id"], doc_id_2)

    def test_rejects_wrong_access_code(self):
        document_id, _ = self._upload_sample_pdf()
        response = self.post_ask(document_id, access_code="wrong")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json["detail"], "Enter the correct access code.")

    def test_requires_ollama_api_key(self):
        with patch.dict(os.environ, {"APP_ACCESS_CODE": "test-access"}, clear=False):
            os.environ.pop("OLLAMA_API_KEY", None)
            document_id, _ = self._upload_sample_pdf()
            # Mock retrieve_chunks to return a result so generate_answer is called
            with patch.object(
                app_module,
                "retrieve_chunks",
                return_value=[
                    {
                        "citation": "test.pdf, page 1",
                        "text": "Some content.",
                        "chunk_id": 1,
                        "similarity": 0.9,
                        "document_id": document_id,
                    }
                ],
            ):
                response = self.post_ask(document_id)
        self.assertEqual(response.status_code, 503)
        self.assertIn("OLLAMA_API_KEY", response.json["detail"])

    def test_rejects_unknown_document_id(self):
        response = self.post_ask("nonexistent-doc-id", "What is this?")
        # Should still return 200 but with no sources / no match
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["sources"], [])

    def test_rejects_missing_document_id(self):
        response = self.client.post(
            "/api/ask",
            data={"question": "What is this?"},
            headers={"X-App-Access-Code": "test-access"},
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("document_id is required", response.json["detail"])

    def test_list_documents(self):
        document_id, _ = self._upload_sample_pdf()
        response = self.client.get(
            "/api/documents",
            headers={"X-App-Access-Code": "test-access"},
        )
        self.assertEqual(response.status_code, 200)
        docs = response.json["documents"]
        self.assertTrue(any(d["document_id"] == document_id for d in docs))

    def test_delete_document(self):
        document_id, _ = self._upload_sample_pdf()
        response = self.client.delete(
            f"/api/documents/{document_id}",
            headers={"X-App-Access-Code": "test-access"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertGreaterEqual(response.json["deleted_chunks"], 1)
        self.assertEqual(response.json["document_id"], document_id)


if __name__ == "__main__":
    unittest.main()
