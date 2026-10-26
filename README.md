# Sourcebook — PDF Chat on Vercel

Sourcebook is a Flask PDF question-answering app with a source sidebar and a scrolling, citation-aware conversation. Attach up to five text-based PDFs, ask questions grounded in those sources, and inspect the page excerpts used for each answer.

## How it works

1. The browser sends the attached PDF sources and question to the Flask API.
2. PyPDF extracts page text and the app creates overlapping, page-aware text chunks.
3. FastEmbed generates chunk and question embeddings with a small CPU model (`BAAI/bge-small-en-v1.5` by default).
4. A LangGraph workflow ranks chunks by cosine similarity and routes to answer generation when relevant passages are found, or to a no-match response otherwise.
5. Ollama Cloud generates an answer from the retrieved excerpts. The UI displays the answer and its source pages.

The embedding model runs on CPU; no GPU is needed. Ollama Cloud is used for chat generation only. Embeddings are computed on each question because Vercel Functions do not provide durable per-user memory or document storage.

## Features and limits

- Up to 5 text-based PDF sources per question, with a combined size limit of 4 MB. Vercel Functions have a 4.5 MB total request-body limit, including multipart form overhead.
- Page-specific citations for PDF excerpts.
- CPU-based embeddings; configure the model using `EMBEDDING_MODEL`.
- LangGraph retrieval and answer/no-match routing.
- Ollama Cloud chat model configured by `OLLAMA_CHAT_MODEL` (default `gemma4:31b`).
- Server-side access-code protection; Vercel deployments fail closed unless `APP_ACCESS_CODE` is configured.
- Source files stay in browser memory and are resent with each question. They are not retained as a notebook or durable vector index.
- Scanned/image-only PDFs are not OCR-processed. DOCX and other formats are not supported by this app.

FastEmbed downloads its selected embedding model when it is first initialized. On Vercel, model files may need to be downloaded again in a new function instance; this can increase cold-start time. A production deployment should use durable source storage and a persistent vector index, or a managed embedding service.

## Run locally

Python 3.12 is recommended. Create a `.env` file from `.env.example`, set a valid Ollama Cloud API key, and install dependencies:

```powershell
Copy-Item .env.example .env
python -m pip install -r requirements.txt
flask --app app run --debug
```

Open `http://127.0.0.1:5000`. The local app does not require an access code unless `APP_ACCESS_CODE` is set. The first question may take longer while FastEmbed downloads the embedding model. Keep `.env` private; never commit API keys.

## Deploy on Vercel

Vercel supports Flask through its Python runtime. The root `app.py` exports a Flask object named `app`; `vercel.json` includes the static UI and excludes tests and sample documents from the function bundle.

1. Push the repository to GitHub and import it as a Vercel project.
2. In Vercel **Settings → Environment Variables**, configure:
   - `OLLAMA_API_KEY`: your Ollama Cloud API key.
   - `APP_ACCESS_CODE`: a long, unique code to restrict access to the model-backed API.
   - `OLLAMA_CHAT_MODEL`: optional; defaults to `gemma4:31b`.
   - `EMBEDDING_MODEL`: optional FastEmbed model name; defaults to `BAAI/bge-small-en-v1.5`.
3. Redeploy after changing environment variables.
4. Open the deployment URL, enter the access code, attach one or more small text-based PDFs, and ask a question.

Confirm that the selected Ollama chat model is available to your account and that your deployment can download the FastEmbed model. The app does not send the Ollama API key to the browser. Do not publish the access code.

## Verify

```powershell
python -m unittest discover -s tests -v
```

Tests use mocked embeddings and Ollama responses; they do not require credentials or make paid inference calls. A live model request and Vercel deployment must be tested separately.

## Project layout

```text
app.py                         Flask routes, LangGraph workflow, Ollama Cloud chat
document_processing.py         PDF extraction, page-aware chunks, vector retrieval
static/index.html              Source sidebar and scrolling chat interface
vercel.json                    Vercel function configuration
requirements.txt               Runtime dependencies
tests/                         API, workflow, and document-processing tests
notebooks/Untitled.ipynb       Separate, non-deployed experiment
documents/                     Sample PDFs, excluded from the Vercel bundle
```
