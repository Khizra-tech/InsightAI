# InsightAI 📄

Chat with, summarize and compare your PDF documents using Retrieval-Augmented Generation (RAG). Answers are grounded in your files and include page-level source citations.

## Features

- Upload multiple PDFs, with text extraction and overlapping chunking
- Local embeddings (sentence-transformers), so no embedding API cost
- Vector search with ChromaDB (cosine similarity)
- Context-grounded Q&A with `[file, p.X]` citations and a sources panel
- Document summarization (map-reduce for long files)
- Side-by-side comparison of two documents on any aspect
- Clean Streamlit interface

Screenshots of the app are in the `screenshots/` folder.

## Architecture

Ingestion:  PDF -> pypdf (text per page) -> chunks (800 chars, 150 overlap)
                -> embeddings (all-MiniLM-L6-v2) -> ChromaDB

Query:      Question -> embedding -> top-k similar chunks
                -> LLM (context + question) -> answer + citations

## Project structure

InsightAI/
├── app/
│   ├── main.py                 # Streamlit UI
│   ├── document_processor.py   # PDF extraction and chunking
│   ├── embeddings.py           # Embedding model wrapper
│   ├── vector_store.py         # ChromaDB wrapper
│   └── rag_pipeline.py         # RAG, summarization, comparison
├── tests/test_rag.py
├── sample_documents/           # Example PDFs to try
├── screenshots/
├── requirements.txt
├── .env.example
└── README.md

## Setup

**Requirements:** Python 3.12 recommended (3.13+ may fail to install some dependencies).

bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 2. Install dependencies
python -m pip install -r requirements.txt

# 3. Configure your API key
copy .env.example .env         # Windows (macOS/Linux: cp .env.example .env)

Open `.env` and set `LLM_API_KEY`. A free key is available at [console.groq.com](https://console.groq.com).

bash
# 4. Run the app
python -m streamlit run app/main.py

Then open http://localhost:8501 and try the PDFs in `sample_documents/`.

## Running the app again

Open a terminal in the project folder, activate the environment and start the app:

bash
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

python -m streamlit run app/main.py

Previously uploaded documents are saved in `chroma_db/`, so you don't need to upload them again. Press Ctrl+C in the terminal to stop the app.

## Configuration

| Variable | Description | Default |
|---|---|---|
| `LLM_API_KEY` | API key for your LLM provider | required |
| `LLM_BASE_URL` | Any OpenAI-compatible endpoint | `https://api.groq.com/openai/v1` |
| `LLM_MODEL` | Chat model name | `openai/gpt-oss-120b` |
| `EMBEDDING_MODEL` | Local embedding model | `all-MiniLM-L6-v2` |

Available models differ by account and change over time. To list the models your key can use:

bash
python -c "import os; from dotenv import load_dotenv; from openai import OpenAI; load_dotenv(); c=OpenAI(api_key=os.getenv('LLM_API_KEY'), base_url=os.getenv('LLM_BASE_URL')); [print(m.id) for m in c.models.list().data]"

## Tests

bash
python -m pytest

## Tech stack

Python, Streamlit, pypdf, sentence-transformers, ChromaDB, OpenAI-compatible LLM API (Groq).

## Troubleshooting

- **PowerShell blocks venv activation:** run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
- **`401 invalid API key`:** make sure `.env` has your real key (not the placeholder) and restart the app.
- **`404 model does not exist`:** set `LLM_MODEL` to a model from the list command above.
- **"No extractable text":** the PDF is probably scanned. OCR is not supported yet.

## Limitations and future improvements

- No OCR for scanned PDFs
- Possible upgrades: hybrid search, re-ranking, evaluation metrics (retrieval accuracy), conversation memory

## Author

Khizra