"""
vector_store.py
================
Embeds paper abstracts with `sentence-transformers` (all-MiniLM-L6-v2, a
free, local 384-dim model -- no API key needed) and stores them in an
embedded ChromaDB collection (persisted to ./data/chroma_db) for semantic
similarity search.

This is the "semantic retrieval" leg of the hybrid recommender / GraphRAG
pipeline: given a free-text research query, it returns the top-K most
semantically similar paper abstracts, independent of the citation/co-author
graph structure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

DATA_DIR = Path(__file__).parent / "data"
CHROMA_PERSIST_DIR = str(DATA_DIR / "chroma_db")
COLLECTION_NAME = "scholarrag_abstracts"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


class VectorStore:
    """Thin wrapper around ChromaDB + SentenceTransformers.

    Lazily imports both heavy dependencies so the rest of the codebase can be
    imported/tested even in environments where they aren't installed yet.
    """

    def __init__(
        self,
        persist_dir: str = CHROMA_PERSIST_DIR,
        collection_name: str = COLLECTION_NAME,
        model_name: str = EMBEDDING_MODEL_NAME,
    ) -> None:
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.model_name = model_name
        self._client = None
        self._collection = None
        self._embedder = None

    # ------------------------------------------------------------------ #
    # Lazy setup
    # ------------------------------------------------------------------ #
    def _get_client(self):
        if self._client is None:
            try:
                import chromadb
            except ImportError as e:
                raise ImportError(
                    "chromadb is not installed. Run `pip install chromadb`."
                ) from e
            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
            # PersistentClient == embedded, on-disk, no server process required.
            self._client = chromadb.PersistentClient(path=self.persist_dir)
        return self._client

    def _get_embedder(self):
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is not installed. "
                    "Run `pip install sentence-transformers`."
                ) from e
            self._embedder = SentenceTransformer(self.model_name)
        return self._embedder

    def _get_collection(self):
        if self._collection is None:
            client = self._get_client()
            # get_or_create keeps this idempotent across app restarts.
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def reset(self) -> None:
        """Deletes the collection so it can be re-indexed cleanly with fresh papers."""
        client = self._get_client()
        try:
            client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self._collection = None

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #
    def index_papers(
        self, data_dir: Path = DATA_DIR, batch_size: int = 32, reset: bool = False
    ) -> None:
        """
        Embeds every paper's abstract and upserts it into the Chroma
        collection, along with lightweight metadata (title, field, year,
        methodology) used to render search results without a second lookup.
        """
        if reset:
            self.reset()

        papers_path = data_dir / "papers.json"
        if not papers_path.exists():
            raise FileNotFoundError(
                "papers.json not found. Run `python openalex_fetcher.py` first."
            )
        papers = json.loads(papers_path.read_text(encoding="utf-8"))

        embedder = self._get_embedder()
        collection = self._get_collection()

        ids = [p["id"] for p in papers]
        documents = [p["abstract"] for p in papers]
        metadatas = [
            {
                "title": p["title"],
                "field": p["field"],
                "methodology": p["methodology"],
                "year": p["year"],
            }
            for p in papers
        ]

        for start in range(0, len(papers), batch_size):
            batch_ids = ids[start:start + batch_size]
            batch_docs = documents[start:start + batch_size]
            batch_meta = metadatas[start:start + batch_size]
            embeddings = embedder.encode(batch_docs, show_progress_bar=False).tolist()
            collection.upsert(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_meta,
                embeddings=embeddings,
            )

        print(f"[vector_store] Indexed {len(papers)} paper abstracts into "
              f"Chroma collection '{self.collection_name}'.")

    # ------------------------------------------------------------------ #
    # Query
    # ------------------------------------------------------------------ #
    def semantic_search(
        self, query: str, top_k: int = 5, field_filter: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """
        Returns the top_k papers whose abstracts are semantically closest to
        `query`, each as {"id", "title", "field", "methodology", "year",
        "abstract", "distance"}. Lower `distance` = more similar (cosine).
        """
        embedder = self._get_embedder()
        collection = self._get_collection()

        query_embedding = embedder.encode([query]).tolist()
        where = {"field": field_filter} if field_filter else None

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=where,
        )

        hits: list[dict[str, Any]] = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for pid, doc, meta, dist in zip(ids, docs, metas, dists):
            hits.append({
                "id": pid,
                "abstract": doc,
                "distance": dist,
                **meta,
            })
        return hits

    def similarity_score(self, query: str, paper_id: str) -> Optional[float]:
        """
        Convenience helper for recommender.py: returns a [0, 1] similarity
        score (1 = identical) between a free-text query and one specific
        paper's abstract, or None if the paper isn't indexed.
        """
        hits = self.semantic_search(query, top_k=50)
        for hit in hits:
            if hit["id"] == paper_id:
                # Cosine distance in [0, 2] -> similarity in [0, 1].
                return max(0.0, 1.0 - hit["distance"] / 2)
        return None


def build_default_vector_store() -> VectorStore:
    """Factory used by app.py / recommender.py: builds (or reuses) the index."""
    store = VectorStore()
    store.index_papers()
    return store


if __name__ == "__main__":
    vs = build_default_vector_store()
    results = vs.semantic_search(
        "graph neural networks for recommendation systems", top_k=5
    )
    print("\nTop semantic matches:")
    for r in results:
        print(f"  [{r['id']}] {r['title']}  (distance={r['distance']:.3f})")
