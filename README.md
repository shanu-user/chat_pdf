# Document QA Assistant

A Streamlit prototype for asking questions about a PDF. It extracts text, splits it into chunks, creates embeddings, retrieves relevant chunks from FAISS, and sends that context to a chat model.

## Current features

- Upload and process one text-based PDF at a time.
- Preserve one-based PDF page numbers and source filenames on indexed chunks.
- Configure chunk size, overlap, and the number of retrieved chunks.
- Ask questions grounded in retrieved document text.
- Review retrieved text excerpts with source filename and page references.
- Keep a question-and-answer history for the current document.
- Replace a document or change chunk settings without accidentally reusing a stale index.

## Source references and limitations

Text is extracted page by page. Each indexed chunk retains its source filename, one-based PDF page number, chunk ID, and character start offset. The app displays the retrieved excerpt with its source filename and page number; references come from retrieved metadata, not model-generated page numbers.

Page references identify the source PDF page from which extracted text came. They are not highlighted quotations or proof that a scanned page was OCR-processed. Scanned/image-only PDFs are rejected with a readable-text message; OCR is not implemented.

## Run locally

Use Python 3.10 or newer. Install the packages in `requirements.txt`, then provide an OpenAI API key through an environment variable or a local `.env` file:

```text
OPENAI_API_KEY=your_api_key
```

Start the app from the project directory:

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

Do not commit `.env` or API keys. For hosted deployments, configure the key using the host's secret-management feature.

## Project layout

```text
app.py                         Streamlit application and RAG workflow
document_processing.py         PDF extraction, page-aware chunking, citations
requirements.txt               Python dependencies
tests/                          Focused document-processing and app-flow tests
documents/                     Sample PDF and DOCX files
notebooks/Untitled.ipynb       Separate PDF/DOCX and Ollama experiment
```

The notebook is separate from the Streamlit app. The app supports PDF upload only; DOCX ingestion and Ollama are not app features. The notebook uses OpenAI embeddings (and therefore needs `OPENAI_API_KEY`) together with a local Ollama server and the `llama3` model. Start Ollama and run `ollama pull llama3` before executing the model/Q&A cells.

## Verify locally

Run the focused tests:

```powershell
python -m unittest discover -s tests -v
```

Then run `streamlit run app.py` and verify a sample PDF, a second replacement PDF, and a blank/image-only PDF. After indexing, change chunk size or overlap and confirm that the app asks you to rebuild the index before answering with it.

## Portfolio presentation

Describe this as a document-question-answering prototype built with Python, Streamlit, LangChain, FAISS, and OpenAI. A transparent demo can show uploading a text PDF, asking a question, and inspecting page-referenced excerpts. Avoid claiming OCR support or highlighted quotations.

Before publishing a live demo, verify the install from a clean environment, test a replacement upload and an unreadable/scanned PDF, and confirm API secrets are configured outside the source code.
