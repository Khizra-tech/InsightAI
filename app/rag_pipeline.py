"""Phases 4 and 5: RAG, summarization, comparison and citations."""
import os
from typing import BinaryIO, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

from document_processor import process_pdf
from embeddings import Embedder
from vector_store import VectorStore

load_dotenv()

QA_SYSTEM = (
    "You are InsightAI, a document assistant. Answer ONLY using the provided context. "
    "Cite sources inline as [filename, p.X]. If the answer is not in the context, say so."
)


def format_context(chunks: List[Dict]) -> str:
    return "\n\n".join(f"[{c['source']}, p.{c['page']}]\n{c['text']}" for c in chunks)


class RAGPipeline:
    def __init__(self, db_path: str = "chroma_db"):
        self.embedder = Embedder()
        self.store = VectorStore(path=db_path)
        self.model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
        self._client = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            key = os.getenv("LLM_API_KEY")
            if not key:
                raise RuntimeError("LLM_API_KEY is missing. Copy .env.example to .env and add your key.")
            self._client = OpenAI(api_key=key, base_url=os.getenv("LLM_BASE_URL"))
        return self._client

    def _chat(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content.strip()

    # ---- Phases 2-3: ingestion ----
    def ingest(self, file: BinaryIO, name: str) -> int:
        chunks = process_pdf(file, name)
        if not chunks:
            raise ValueError(f"No extractable text in {name} (scanned PDF?).")
        embeddings = self.embedder.embed([c["text"] for c in chunks])
        self.store.add_chunks(chunks, embeddings)
        return len(chunks)

    # ---- Phase 4: RAG ----
    def answer(self, question: str, k: int = 5, source: str = None) -> Dict:
        q_emb = self.embedder.embed([question])[0]
        chunks = self.store.query(q_emb, k=k, source=source)
        if not chunks:
            return {"answer": "No documents uploaded yet.", "sources": []}
        prompt = f"Context:\n{format_context(chunks)}\n\nQuestion: {question}"
        return {"answer": self._chat(QA_SYSTEM, prompt), "sources": chunks}

    # ---- Phase 5: AI features ----
    def summarize(self, source: str, batch_chars: int = 6000, max_batches: int = 8) -> str:
        chunks = self.store.get_document_chunks(source)
        if not chunks:
            return "Document not found."
        batches, current = [], ""
        for c in chunks:
            if len(current) + len(c["text"]) > batch_chars and current:
                batches.append(current)
                current = ""
            current += c["text"] + "\n"
        batches.append(current)
        batches = batches[:max_batches]
        partials = [
            self._chat("Summarize this section of a document concisely.", b) for b in batches
        ]
        if len(partials) == 1:
            return partials[0]
        return self._chat(
            "Combine these partial summaries into one clear summary with a short overview "
            "followed by key points.",
            "\n\n".join(partials),
        )

    def compare(self, source_a: str, source_b: str, aspect: str = "main topics and conclusions", k: int = 4) -> str:
        q_emb = self.embedder.embed([aspect])[0]
        a = self.store.query(q_emb, k=k, source=source_a)
        b = self.store.query(q_emb, k=k, source=source_b)
        prompt = (
            f"Compare the two documents on: {aspect}\n\n"
            f"=== Document A: {source_a} ===\n{format_context(a)}\n\n"
            f"=== Document B: {source_b} ===\n{format_context(b)}"
        )
        return self._chat(
            "You compare documents. Give similarities, differences and a short conclusion. "
            "Cite sources inline as [filename, p.X]. Use only the provided context.",
            prompt,
        )
