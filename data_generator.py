"""
data_generator.py
==================
Generates a realistic, internally-consistent mock academic dataset for
ScholarRAG: papers, authors, institutions, fields of study, methodologies,
citation edges, authorship edges, and synthetic user reading/bookmarking
logs (used later by the ALS recommender).

The data is deterministic (seeded) so every module downstream can be
developed/tested against the same graph. Output is written to ./data/*.json
and also returned as pandas DataFrames for in-memory pipelines.

Run directly:
    python data_generator.py
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

import pandas as pd

RANDOM_SEED = 42
DATA_DIR = Path(__file__).parent / "data"


# --------------------------------------------------------------------------- #
# Domain vocabulary used to make synthetic abstracts read naturally
# --------------------------------------------------------------------------- #

FIELDS_OF_STUDY = [
    "Natural Language Processing",
    "Computer Vision",
    "Graph Neural Networks",
    "Reinforcement Learning",
    "Information Retrieval",
    "Knowledge Graphs",
    "Recommender Systems",
    "Federated Learning",
]

METHODOLOGIES = [
    "Transformer Architectures",
    "Contrastive Learning",
    "Graph Attention Networks",
    "Matrix Factorization",
    "Retrieval-Augmented Generation",
    "Self-Supervised Pretraining",
    "Diffusion Models",
    "Reinforcement Learning from Human Feedback",
]

INSTITUTIONS = [
    "Stanford University",
    "MIT",
    "University of Toronto",
    "ETH Zurich",
    "Carnegie Mellon University",
    "National University of Singapore",
    "University of Cambridge",
    "UC Berkeley",
]

FIRST_NAMES = [
    "Ava", "Liam", "Maya", "Noah", "Priya", "Ethan", "Sofia", "Kenji",
    "Zara", "Omar", "Elena", "Wei", "Ines", "Lucas", "Nadia", "Sam",
]
LAST_NAMES = [
    "Chen", "Okafor", "Rossi", "Kumar", "Nakamura", "Novak", "Silva",
    "Haddad", "Fischer", "Park", "Duran", "Petrov", "Adeyemi", "Larsson",
]

ABSTRACT_TEMPLATES = [
    "We propose a novel approach to {field} that leverages {method1} to "
    "improve downstream performance. Our method combines {method2} with a "
    "carefully designed training objective, achieving state-of-the-art "
    "results on standard benchmarks while reducing computational overhead.",

    "This paper investigates the intersection of {method1} and {field}, "
    "introducing a framework that unifies representation learning with "
    "task-specific fine-tuning. Extensive experiments show that our method "
    "outperforms prior work based on {method2} by a significant margin.",

    "Recent advances in {field} have been driven largely by {method1}. In "
    "this work, we identify key limitations of existing approaches and "
    "propose a hybrid architecture that integrates {method2}, yielding "
    "improved generalization and sample efficiency.",

    "We present a large-scale empirical study of {method1} applied to "
    "{field}. Our analysis reveals that combining it with {method2} "
    "substantially improves robustness under distribution shift, with "
    "implications for real-world deployment.",
]

TITLE_TEMPLATES = [
    "Rethinking {method1} for {field}",
    "{method1} Meets {method2}: A Unified Framework for {field}",
    "Towards Robust {field} via {method1}",
    "Scaling {method1} for Real-World {field} Applications",
    "A Survey of {method1} Techniques in {field}",
]


@dataclass
class Author:
    id: str
    name: str
    institution: str


@dataclass
class Paper:
    id: str
    title: str
    abstract: str
    year: int
    field: str
    methodology: str
    author_ids: list[str] = field(default_factory=list)
    citation_ids: list[str] = field(default_factory=list)  # papers THIS paper cites


def _generate_authors(n: int, rng: random.Random) -> list[Author]:
    authors = []
    used_names: set[str] = set()
    for i in range(n):
        while True:
            name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
            if name not in used_names:
                used_names.add(name)
                break
        authors.append(
            Author(id=f"A{i:03d}", name=name, institution=rng.choice(INSTITUTIONS))
        )
    return authors


def _generate_papers(
    n: int, authors: list[Author], rng: random.Random
) -> list[Paper]:
    papers: list[Paper] = []
    for i in range(n):
        field_choice = rng.choice(FIELDS_OF_STUDY)
        method1 = rng.choice(METHODOLOGIES)
        method2 = rng.choice([m for m in METHODOLOGIES if m != method1])

        title = rng.choice(TITLE_TEMPLATES).format(
            field=field_choice, method1=method1, method2=method2
        )
        abstract = rng.choice(ABSTRACT_TEMPLATES).format(
            field=field_choice, method1=method1, method2=method2
        )
        year = rng.randint(2018, 2024)

        # 1-3 authors per paper, biased toward 1-2
        n_authors = rng.choices([1, 2, 3], weights=[0.4, 0.4, 0.2])[0]
        paper_authors = rng.sample(authors, k=min(n_authors, len(authors)))

        papers.append(
            Paper(
                id=f"P{i:03d}",
                title=title,
                abstract=abstract,
                year=year,
                field=field_choice,
                methodology=method1,
                author_ids=[a.id for a in paper_authors],
            )
        )
    return papers


def _wire_citations(papers: list[Paper], rng: random.Random, max_refs: int = 4) -> None:
    """
    Papers may only cite earlier (lower-index / same-or-earlier-year) papers,
    which keeps the citation DAG acyclic -- required for meaningful citation-
    depth and multi-hop traversal queries later in graph_store.py.
    """
    papers_sorted = sorted(papers, key=lambda p: (p.year, p.id))
    for idx, paper in enumerate(papers_sorted):
        if idx == 0:
            continue
        candidates = papers_sorted[:idx]
        k = min(rng.randint(0, max_refs), len(candidates))
        if k == 0:
            continue
        # Prefer citing papers in the same field slightly more often
        same_field = [p for p in candidates if p.field == paper.field]
        pool = same_field if same_field and rng.random() < 0.6 else candidates
        k = min(k, len(pool))
        cited = rng.sample(pool, k=k)
        paper.citation_ids = [p.id for p in cited]


def _generate_reading_logs(
    papers: list[Paper], authors: list[Author], rng: random.Random, n_users: int = 15
) -> list[dict[str, Any]]:
    """
    Simulates implicit feedback (views / bookmarks) from n_users synthetic
    researcher-users reading papers. Each interaction has a weight
    representing implicit confidence (view=1, bookmark=3, cite-in-draft=5),
    which is exactly the kind of signal `implicit`'s ALS expects.
    """
    users = [f"U{i:03d}" for i in range(n_users)]
    action_weights = {"view": 1, "bookmark": 3, "cited_in_draft": 5}
    logs = []
    for user in users:
        # Each synthetic user has a topical bias (mimics a real researcher's niche)
        preferred_field = rng.choice(FIELDS_OF_STUDY)
        candidates = [p for p in papers if p.field == preferred_field] or papers
        n_interactions = rng.randint(3, 10)
        interacted = rng.sample(candidates, k=min(n_interactions, len(candidates)))
        for paper in interacted:
            action = rng.choices(
                list(action_weights.keys()), weights=[0.6, 0.3, 0.1]
            )[0]
            logs.append(
                {
                    "user_id": user,
                    "paper_id": paper.id,
                    "action": action,
                    "weight": action_weights[action],
                }
            )
    return logs


def generate_dataset(
    n_authors: int = 18,
    n_papers: int = 30,
    n_users: int = 15,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Generates the full mock dataset and returns it as plain dicts/lists."""
    rng = random.Random(seed)
    authors = _generate_authors(n_authors, rng)
    papers = _generate_papers(n_papers, authors, rng)
    _wire_citations(papers, rng)
    reading_logs = _generate_reading_logs(papers, authors, rng, n_users)

    return {
        "authors": [asdict(a) for a in authors],
        "papers": [asdict(p) for p in papers],
        "fields_of_study": FIELDS_OF_STUDY,
        "methodologies": METHODOLOGIES,
        "institutions": INSTITUTIONS,
        "reading_logs": reading_logs,
    }


def save_dataset(dataset: dict[str, Any], out_dir: Path = DATA_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, value in dataset.items():
        with open(out_dir / f"{key}.json", "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2)
    print(f"[data_generator] Wrote dataset to {out_dir.resolve()}")


def as_dataframes(dataset: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Convenience accessor for modules that prefer DataFrames over dicts."""
    return {
        "authors": pd.DataFrame(dataset["authors"]),
        "papers": pd.DataFrame(dataset["papers"]),
        "reading_logs": pd.DataFrame(dataset["reading_logs"]),
    }


if __name__ == "__main__":
    import sys
    if "--mock" in sys.argv:
        print("[data_generator] Generating synthetic mock dataset...")
        ds = generate_dataset()
        save_dataset(ds)
    else:
        try:
            from openalex_fetcher import build_real_academic_dataset, save_real_dataset
            print("[data_generator] Ingesting real academic papers & authors from OpenAlex API...")
            ds = build_real_academic_dataset()
            save_real_dataset(ds)
        except Exception as e:
            print(f"[data_generator] External API fetch encountered {e}. Falling back to synthetic generator...")
            ds = generate_dataset()
            save_dataset(ds)

    dfs = as_dataframes(ds)
    print(f"Authors: {len(dfs['authors'])} | Papers: {len(dfs['papers'])} "
          f"| Reading-log events: {len(dfs['reading_logs'])}")
