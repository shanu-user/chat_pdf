import os
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch

import app as application


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PDF = (
    PROJECT_ROOT / "documents" / "Why-We-Use-Common-Paper-Standard-Agreements.pdf"
)
SAMPLE_BYTES = SAMPLE_PDF.read_bytes()


class AppTests(unittest.TestCase):
    def setUp(self):
        self.client = application.app.test_client()
        self.env_patch = patch.dict(
            os.environ,
            {"OLLAMA_API_KEY": "test-api-key", "APP_ACCESS_CODE": "test-access"},
            clear=False,
        )
        self.env_patch.start()

    def tearDown(self):
        self.env_patch.stop()

    def post_sources(
        self,
        sources=None,
        question="What are the standard agreements?",
        access_code="test-access",
    ):
        data = {
            "question": question,
            "sources": [
                (BytesIO(content), filename)
                for filename, content in (sources or [(SAMPLE_PDF.name, SAMPLE_BYTES)])
            ],
        }
        headers = {}
        if access_code is not None:
            headers["X-App-Access-Code"] = access_code
        return self.client.post("/api/ask", data=data, headers=headers)

    def test_health_and_frontend_routes(self):
        health = self.client.get("/api/health")
        frontend = self.client.get("/")

        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json, {"status": "ok"})
        self.assertEqual(frontend.status_code, 200)
        self.assertIn(b"Sourcebook", frontend.data)
        self.assertIn(b"Add sources", frontend.data)
        self.assertIn(b"conversation.scrollTop", frontend.data)
        health.close()
        frontend.close()

    def test_ask_returns_answer_and_page_cited_sources(self):
        def fake_embeddings(texts):
            return [[1.0, 0.0] for _ in texts]

        with (
            patch.object(application, "embed_texts", side_effect=fake_embeddings),
            patch.object(
                application,
                "generate_answer",
                return_value="The document describes standard agreements.",
            ),
        ):
            response = self.post_sources()

        self.assertEqual(response.status_code, 200, response.data)
        self.assertEqual(
            response.json["answer"],
            "The document describes standard agreements.",
        )
        self.assertTrue(response.json["sources"])
        self.assertTrue(
            all(
                source["citation"].endswith(", page 1")
                for source in response.json["sources"]
            )
        )

    def test_graph_routes_to_no_match_without_calling_chat_model(self):
        def unrelated_embeddings(texts):
            return [[0.0, 1.0] for _ in texts[:-1]] + [[1.0, 0.0]]

        with (
            patch.object(application, "embed_texts", side_effect=unrelated_embeddings),
            patch.object(application, "generate_answer") as generate,
        ):
            response = self.post_sources(question="A completely unrelated question")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["sources"], [])
        self.assertIn("couldn't find relevant passages", response.json["answer"])
        generate.assert_not_called()

    def test_ollama_chat_uses_server_side_bearer_auth(self):
        response = Mock()
        response.json.return_value = {"message": {"content": "A grounded answer."}}
        with patch.object(application.httpx, "post", return_value=response) as post:
            answer = application.generate_answer(
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
        self.assertEqual(request.kwargs["json"]["model"], application.CHAT_MODEL)
        self.assertIn(
            "agreement.pdf, page 2",
            request.kwargs["json"]["messages"][1]["content"],
        )

    def test_supports_multiple_source_files(self):
        def fake_embeddings(texts):
            return [[1.0, 0.0] for _ in texts]

        second_name = "second-agreement.pdf"
        with (
            patch.object(application, "embed_texts", side_effect=fake_embeddings),
            patch.object(application, "generate_answer", return_value="Grounded answer."),
        ):
            response = self.post_sources(
                sources=[(SAMPLE_PDF.name, SAMPLE_BYTES), (second_name, SAMPLE_BYTES)]
            )

        self.assertEqual(response.status_code, 200, response.data)
        citations = {source["citation"].split(", page")[0] for source in response.json["sources"]}
        self.assertEqual(citations, {SAMPLE_PDF.name, second_name})

    def test_rejects_wrong_access_code(self):
        response = self.post_sources(access_code="wrong")

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json["detail"], "Enter the correct access code.")

    def test_vercel_requires_access_code_configuration(self):
        with patch.dict(os.environ, {"VERCEL": "1"}, clear=False):
            os.environ.pop("APP_ACCESS_CODE", None)
            response = self.post_sources(access_code=None)

        self.assertEqual(response.status_code, 503)
        self.assertIn("Configure APP_ACCESS_CODE", response.json["detail"])

    def test_requires_ollama_api_key_only_when_retrieval_finds_sources(self):
        with patch.dict(os.environ, {"APP_ACCESS_CODE": "test-access"}, clear=False):
            os.environ.pop("OLLAMA_API_KEY", None)
            with patch.object(
                application,
                "embed_texts",
                side_effect=lambda texts: [[1.0, 0.0] for _ in texts],
            ):
                response = self.post_sources()

        self.assertEqual(response.status_code, 503)
        self.assertIn("OLLAMA_API_KEY", response.json["detail"])

    def test_rejects_non_pdf_upload(self):
        response = self.post_sources(sources=[("notes.txt", b"plain text")])

        self.assertEqual(response.status_code, 415)

    def test_rejects_blank_pdf(self):
        from pypdf import PdfWriter

        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        buffer = BytesIO()
        writer.write(buffer)

        response = self.post_sources(sources=[("blank.pdf", buffer.getvalue())])

        self.assertEqual(response.status_code, 422)
        self.assertIn("No readable text", response.json["detail"])

    def test_rejects_combined_upload_larger_than_limit(self):
        too_large = b"%PDF-" + b"x" * application.MAX_SOURCE_BYTES
        response = self.post_sources(sources=[("large.pdf", too_large)])

        self.assertEqual(response.status_code, 413)
        self.assertIn("4 MB combined", response.json["detail"])

    def test_rejects_duplicate_source_names(self):
        response = self.post_sources(
            sources=[("duplicate.pdf", SAMPLE_BYTES), ("duplicate.pdf", SAMPLE_BYTES)]
        )

        self.assertEqual(response.status_code, 422)
        self.assertIn("different filename", response.json["detail"])


if __name__ == "__main__":
    unittest.main()
