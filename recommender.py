"""
recommender.py
================
Hybrid recommendation engine combining three independent signals into one
ranked list of papers for a given user + optional query:

    1. Collaborative Filtering (Implicit ALS)
       Learned from synthetic reading/bookmark/cite-in-draft logs. Captures
       "users like you also read this" patterns the graph and vector
       signals cannot see.

    2. Graph Centrality
       In-degree citation centrality from graph_store.py, as a proxy for a
       paper's influence/importance in the literature.

    3. Semantic Vector Similarity
       Cosine similarity between the user's free-text query and each
       paper's abstract embedding (vector_store.py). Captures topical
       relevance even for papers the user has never interacted with.

Final score is a weighted blend, each component min-max normalized to
[0, 1] first so no single signal dominates purely due to differing scales.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from graph_store import BaseGraphStore
from vector_store import VectorStore


@dataclass
class RecommendationWeights:
    """Blend weights for the three signals. Must sum to 1.0 (not enforced,
    but recommended for interpretability of the final score)."""
    collaborative: float = 0.4
    graph_centrality: float = 0.25
    semantic: float = 0.35


@dataclass
class ScoredPaper:
    paper_id: str
    total_score: float
    collaborative_score: float
    graph_score: float
    semantic_score: float


class HybridRecommender:
    def __init__(
        self,
        graph_store: BaseGraphStore,
        vector_store: VectorStore,
        weights: Optional[RecommendationWeights] = None,
    ) -> None:
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.weights = weights or RecommendationWeights()

        self._als_model = None
        self._user_ids: list[str] = []
        self._paper_ids: list[str] = []
        self._user_index: dict[str, int] = {}
        self._paper_index: dict[str, int] = {}
        self._user_item_matrix: Optional[csr_matrix] = None

    # ------------------------------------------------------------------ #
    # Stage 1: Collaborative filtering via Implicit ALS
    # ------------------------------------------------------------------ #
    def fit_collaborative_model(
        self, reading_logs: pd.DataFrame, factors: int = 16, iterations: int = 20
    ) -> None:
        """
        Trains an ALS (Alternating Least Squares) matrix factorization model
        on implicit feedback (view/bookmark/cite weights) using the
        `implicit` library, which is purpose-built for exactly this kind of
        confidence-weighted implicit-feedback data (as opposed to explicit
        1-5 star ratings, which we don't have).
        """
        try:
            import implicit
        except ImportError as e:
            raise ImportError(
                "`implicit` is not installed. Run `pip install implicit`."
            ) from e

        self._user_ids = sorted(reading_logs["user_id"].unique())
        self._paper_ids = sorted(reading_logs["paper_id"].unique())
        self._user_index = {u: i for i, u in enumerate(self._user_ids)}
        self._paper_index = {p: i for i, p in enumerate(self._paper_ids)}

        rows = reading_logs["user_id"].map(self._user_index)
        cols = reading_logs["paper_id"].map(self._paper_index)
        values = reading_logs["weight"].astype(float)

        # `implicit` expects a (users x items) sparse confidence matrix.
        self._user_item_matrix = csr_matrix(
            (values, (rows, cols)),
            shape=(len(self._user_ids), len(self._paper_ids)),
        )

        self._als_model = implicit.als.AlternatingLeastSquares(
            factors=factors, iterations=iterations, regularization=0.01
        )
        # implicit>=0.6 fits on a (users, items) matrix directly.
        self._als_model.fit(self._user_item_matrix)
        print(f"[recommender] Fit ALS on {len(self._user_ids)} users x "
              f"{len(self._paper_ids)} papers.")

    def _collaborative_scores(self, user_id: str) -> dict[str, float]:
        """Returns raw ALS preference scores for every paper, for one user.
        Unseen users (cold start) get all-zero scores, which the blend step
        then naturally defers to the graph + semantic signals."""
        if self._als_model is None or user_id not in self._user_index:
            return {}
        user_idx = self._user_index[user_id]
        item_ids, scores = self._als_model.recommend(
            user_idx,
            self._user_item_matrix[user_idx],
            N=len(self._paper_ids),
            filter_already_liked_items=False,
        )
        idx_to_paper = {i: p for p, i in self._paper_index.items()}
        return {idx_to_paper[i]: float(s) for i, s in zip(item_ids, scores)}

    # ------------------------------------------------------------------ #
    # Stage 2 + 3: graph centrality & semantic similarity
    # ------------------------------------------------------------------ #
    def _graph_scores(self) -> dict[str, float]:
        return self.graph_store.degree_centrality()

    def _semantic_scores(self, query: str, candidate_ids: list[str]) -> dict[str, float]:
        if not query:
            return {}
        hits = self.vector_store.semantic_search(query, top_k=len(candidate_ids) or 50)
        # cosine distance in [0, 2] -> similarity in [0, 1] (higher = better)
        return {h["id"]: max(0.0, 1.0 - h["distance"] / 2) for h in hits}

    # ------------------------------------------------------------------ #
    # Blending
    # ------------------------------------------------------------------ #
    @staticmethod
    def _min_max_normalize(scores: dict[str, float], all_ids: list[str]) -> dict[str, float]:
        values = [scores.get(pid, 0.0) for pid in all_ids]
        lo, hi = min(values, default=0.0), max(values, default=0.0)
        if hi - lo < 1e-9:
            return {pid: 0.0 for pid in all_ids}
        return {pid: (scores.get(pid, 0.0) - lo) / (hi - lo) for pid in all_ids}

    def recommend(
        self,
        user_id: Optional[str] = None,
        query: Optional[str] = None,
        top_k: int = 10,
    ) -> list[ScoredPaper]:
        """
        Produces a ranked list blending all three signals. Either `user_id`,
        `query`, or both may be supplied:
            - user_id only  -> pure collaborative + graph recommendation
            - query only    -> pure content-based (semantic + graph) search
            - both          -> full hybrid personalized + query-aware ranking
        """
        all_ids = self.graph_store.all_paper_ids()

        raw_collab = self._collaborative_scores(user_id) if user_id else {}
        raw_graph = self._graph_scores()
        raw_semantic = self._semantic_scores(query, all_ids) if query else {}

        norm_collab = self._min_max_normalize(raw_collab, all_ids)
        norm_graph = self._min_max_normalize(raw_graph, all_ids)
        norm_semantic = self._min_max_normalize(raw_semantic, all_ids)

        w = self.weights
        # If a signal wasn't requested (no query / cold-start user), its
        # weight is redistributed proportionally across the remaining signals
        # so the score doesn't get artificially deflated.
        active_weights = {
            "collaborative": w.collaborative if raw_collab else 0.0,
            "graph_centrality": w.graph_centrality,
            "semantic": w.semantic if raw_semantic else 0.0,
        }
        total_active = sum(active_weights.values()) or 1.0
        active_weights = {k: v / total_active for k, v in active_weights.items()}

        scored: list[ScoredPaper] = []
        for pid in all_ids:
            c, g, s = norm_collab[pid], norm_graph[pid], norm_semantic[pid]
            total = (
                active_weights["collaborative"] * c
                + active_weights["graph_centrality"] * g
                + active_weights["semantic"] * s
            )
            scored.append(ScoredPaper(pid, total, c, g, s))

        scored.sort(key=lambda x: x.total_score, reverse=True)
        return scored[:top_k]


if __name__ == "__main__":
    import json
    from pathlib import Path
    from graph_store import get_default_graph_store
    from vector_store import build_default_vector_store

    logs = pd.DataFrame(json.loads((Path("data") / "reading_logs.json").read_text()))
    gs = get_default_graph_store()
    vs = build_default_vector_store()

    rec = HybridRecommender(gs, vs)
    rec.fit_collaborative_model(logs)

    sample_user = logs["user_id"].iloc[0]
    results = rec.recommend(
        user_id=sample_user,
        query="retrieval augmented generation for scientific literature",
        top_k=5,
    )
    print(f"\nTop recommendations for {sample_user}:")
    for r in results:
        print(f"  {r.paper_id}: total={r.total_score:.3f} "
              f"(collab={r.collaborative_score:.2f}, graph={r.graph_score:.2f}, "
              f"semantic={r.semantic_score:.2f})")
