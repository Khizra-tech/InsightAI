"""Phase 3b: ChromaDB wrapper (persistent, cosine similarity)."""
from typing import Dict, List, Optional

import chromadb


class VectorStore:
    def __init__(self, path: str = "chroma_db", name: str = "insightai"):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: List[Dict], embeddings: List[List[float]]) -> None:
        if not chunks:
            return
        self.collection.upsert(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            embeddings=embeddings,
            metadatas=[{"source": c["source"], "page": c["page"]} for c in chunks],
        )

    def query(self, embedding: List[float], k: int = 5, source: Optional[str] = None) -> List[Dict]:
        count = self.collection.count()
        if count == 0:
            return []
        res = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(k, count),
            where={"source": source} if source else None,
        )
        results = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            results.append(
                {"text": doc, "source": meta["source"], "page": meta["page"], "score": round(1 - dist, 3)}
            )
        return results

    def get_document_chunks(self, source: str) -> List[Dict]:
        res = self.collection.get(where={"source": source})
        items = [
            {"text": d, "source": m["source"], "page": m["page"]}
            for d, m in zip(res["documents"], res["metadatas"])
        ]
        return sorted(items, key=lambda x: x["page"])

    def list_documents(self) -> List[str]:
        res = self.collection.get(include=["metadatas"])
        return sorted({m["source"] for m in res["metadatas"]})

    def delete_document(self, source: str) -> None:
        self.collection.delete(where={"source": source})
