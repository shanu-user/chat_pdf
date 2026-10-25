import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import langchain_openai
from langchain_core.documents import Document
from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class FakeVectorStore:
    def __init__(self):
        self.queries = []
        self.documents = [
            Document(
                page_content="Termination requires written notice.",
                metadata={
                    "source": "agreement.pdf",
                    "page_number": 3,
                    "chunk_id": 1,
                },
            )
        ]

    def similarity_search(self, query, k):
        self.queries.append((query, k))
        return self.documents[:k]


class FakeChatOpenAI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def invoke(self, prompt):
        return SimpleNamespace(content="Termination requires written notice.")


class AppTests(unittest.TestCase):
    def create_ready_app(self):
        app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
        state = app.session_state
        vector_store = FakeVectorStore()
        state["active_document_hash"] = None
        state["document_name"] = "agreement.pdf"
        state["pages"] = [
            {"source": "agreement.pdf", "page_number": 3, "text": "Agreement text"}
        ]
        state["chunks"] = vector_store.documents
        state["vector_store"] = vector_store
        state["indexed_chunk_settings"] = (1000, 200)
        state["pipeline_status"] = {
            "ingest": "done",
            "chunk": "done",
            "embed": "done",
            "ready": True,
        }
        app.run()
        self.assertFalse(app.exception, [str(exc.value) for exc in app.exception])
        return app, vector_store

    def test_question_uses_retrieval_and_renders_page_reference(self):
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch.object(langchain_openai, "ChatOpenAI", FakeChatOpenAI),
        ):
            app, vector_store = self.create_ready_app()
            app.text_input(key="question_input").set_value("What is the termination notice?")
            app.run()
            ask_button = next(button for button in app.button if button.label == "Ask")
            ask_button.click().run()

        self.assertFalse(app.exception, [str(exc.value) for exc in app.exception])
        self.assertEqual(vector_store.queries, [("What is the termination notice?", 3)])
        self.assertIn(
            ("bot", "Termination requires written notice.", vector_store.documents),
            app.session_state["chat_history"],
        )
        self.assertTrue(
            any("agreement.pdf, page 3" in element.value for element in app.markdown)
        )

    def test_chunk_setting_change_invalidates_index(self):
        app, _ = self.create_ready_app()

        app.slider[0].set_value(1100).run()

        self.assertFalse(app.exception, [str(exc.value) for exc in app.exception])
        self.assertIsNone(app.session_state["vector_store"])
        self.assertEqual(
            app.session_state["pipeline_status"],
            {
                "ingest": "done",
                "chunk": "pending",
                "embed": "pending",
                "ready": False,
            },
        )


if __name__ == "__main__":
    unittest.main()
