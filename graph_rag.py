"""
graph_rag.py
=============
The GraphRAG pipeline: this is where "graph" and "RAG" actually meet.

Standard RAG retrieves top-K semantically similar chunks and stuffs them
into a prompt. GraphRAG additionally walks the knowledge graph *around*
those retrieved chunks -- citation chains, shared methodologies,
co-authorship links -- and folds that structured context into the prompt
too. This lets the LLM explain not just "here are relevant papers" but
*why* they're connected: "Paper A cites Paper B, whose author Dr. X
co-authored with Dr. Y on a related methodology."

Pipeline stages:
    1. Vector retrieval   -> top-K abstracts semantically matching the query
    2. Graph expansion    -> for each retrieved paper, pull citation chains,
                              shared-methodology papers, and author
                              co-authorship links from graph_store.py
    3. Context assembly   -> merge both into a single structured prompt
    4. LLM synthesis      -> Groq (Llama) or Gemini generates a plain-English
                              literature review + collaboration explanation

LLM provider is pluggable via `LLMProvider`; both Groq and Gemini free
tiers are supported. If no API key is configured, `graph_rag.py` falls back
to a deterministic, template-based summary so the rest of the app remains
demoable offline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal, Optional

from dotenv import load_dotenv

# Load .env file automatically
load_dotenv(override=True)

from graph_store import BaseGraphStore
from vector_store import VectorStore

LLMProviderName = Literal["groq", "gemini", "offline"]


@dataclass
class GraphRAGContext:
    """Structured intermediate representation passed to the LLM prompt --
    kept as a dataclass (rather than raw text) so app.py can also render it
    directly in the UI (e.g. as the pyvis graph) without re-deriving it."""
    query: str
    retrieved_papers: list[dict[str, Any]]
    citation_paths: dict[str, list[list[str]]]          # paper_id -> chains
    shared_methodology: dict[str, list[str]]              # paper_id -> related paper ids
    coauthorship_links: list[tuple[str, str, list[str]]]  # (author_a, author_b, path)


GROQ_DEFAULT_MODELS = [
    "qwen/qwen3.8-27b",
    "groq/compound",
    "openai/gpt-oss-120b",
    "qwen/qwen3.6-27b",
    "openai/gpt-oss-20b",
]


class LLMProvider:
    """Wraps whichever free-tier LLM API is configured via environment
    variables, with a safe offline fallback so the pipeline never hard-fails
    just because no API key is present."""

    def __init__(self, provider: LLMProviderName = "groq", model: Optional[str] = None) -> None:
        self.provider = provider
        self.model = model or ("qwen/qwen3.8-27b" if provider == "groq" else "gemini-2.0-flash")

    def generate(self, prompt: str, system_prompt: str, max_tokens: int = 1000) -> str:
        if self.provider == "groq":
            return self._generate_groq(prompt, system_prompt, max_tokens)
        if self.provider == "gemini":
            return self._generate_gemini(prompt, system_prompt, max_tokens)
        return self._generate_offline(prompt)

    def _generate_groq(self, prompt: str, system_prompt: str, max_tokens: int) -> str:
        load_dotenv(override=True)
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key or not api_key.strip():
            return self._generate_offline(
                prompt,
                note="(GROQ_API_KEY not found in .env or environment — running in offline preview mode)"
            )
        try:
            from groq import Groq
            client = Groq(api_key=api_key.strip())
            
            # Candidate models to try in order
            candidate_models = [self.model] + [m for m in GROQ_DEFAULT_MODELS if m != self.model]
            last_err = None

            for model_name in candidate_models:
                try:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=max_tokens,
                        temperature=0.3,
                    )
                    content = response.choices[0].message.content or ""
                    if content.strip():
                        return content
                except Exception as model_err:
                    last_err = model_err
                    continue

            if last_err:
                raise last_err
            return self._generate_offline(prompt, note="(Groq returned empty response)")
        except Exception as e:  # noqa: BLE001 - surface any API/network error safely
            return self._generate_offline(prompt, note=f"(Groq API error: {e})")

    def _generate_gemini(self, prompt: str, system_prompt: str, max_tokens: int) -> str:
        load_dotenv(override=True)
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key or not api_key.strip():
            return self._generate_offline(
                prompt,
                note="(GEMINI_API_KEY not found in .env or environment — running in offline preview mode)"
            )
        try:
            from google import genai
            client = genai.Client(api_key=api_key.strip())
            response = client.models.generate_content(
                model=self.model,
                contents=f"{system_prompt}\n\n{prompt}",
            )
            return response.text or ""
        except Exception as e:  # noqa: BLE001
            return self._generate_offline(prompt, note=f"(Gemini API error: {e})")

    @staticmethod
    def _generate_offline(prompt: str, note: str = "(offline template mode)") -> str:
        """Deterministic fallback: never blocks the demo on a missing API key."""
        return (
            f"> [!NOTE]\n"
            f"> **LLM Synthesis Status**: {note}\n\n"
            "### 📚 GraphRAG Literature Review (Offline Synthesis)\n\n"
            "**Key Findings & Thematic Overview**:\n"
            "The retrieved papers form a cohesive cluster centered around Graph Neural Networks, "
            "Graph Attention mechanisms, and scalable representation learning. Multiple works converge on "
            "addressing robustness, sample efficiency, and scalability limitations in graph-based recommendation.\n\n"
            "**Citation & Lineage Dynamics**:\n"
            "Traversal of the citation graph reveals strong evolutionary lineage. Foundational architectures "
            "established in earlier papers provide the structural representation baselines, while subsequent works "
            "extend them using Self-Supervised Pretraining and Diffusion objectives.\n\n"
            "**Collaboration & Community Bridges**:\n"
            "The co-authorship network reveals key collaborative bridges between researchers across institutions, "
            "suggesting active cross-pollination between recommendation systems and graph representation learning groups.\n\n"
            "*To enable live LLM synthesis via Groq, add your `GROQ_API_KEY` into the `.env` file or use the sidebar key configuration.*"
        )


class GraphRAGPipeline:
    def __init__(
        self,
        graph_store: BaseGraphStore,
        vector_store: VectorStore,
        llm_provider: Optional[LLMProvider] = None,
    ) -> None:
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.llm = llm_provider or LLMProvider(provider="groq")

    # ------------------------------------------------------------------ #
    # Stage 1: vector retrieval
    # ------------------------------------------------------------------ #
    def _retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return self.vector_store.semantic_search(query, top_k=top_k)

    # ------------------------------------------------------------------ #
    # Stage 2: graph expansion around retrieved papers
    # ------------------------------------------------------------------ #
    def _expand_graph_context(
        self, retrieved_papers: list[dict[str, Any]]
    ) -> tuple[dict[str, list[list[str]]], dict[str, list[str]], list[tuple[str, str, list[str]]]]:
        citation_paths: dict[str, list[list[str]]] = {}
        shared_methodology: dict[str, list[str]] = {}
        coauthorship_links: list[tuple[str, str, list[str]]] = []

        retrieved_ids = [p["id"] for p in retrieved_papers]

        for paper_id in retrieved_ids:
            # Multi-hop citation traversal: "who does this paper build on?"
            citation_paths[paper_id] = self.graph_store.citation_chain(paper_id, max_depth=2)
            # Methodological neighbors: "who else uses this same technique?"
            shared_methodology[paper_id] = self.graph_store.shared_methodology_papers(paper_id)[:3]

        # Co-authorship: for every pair of retrieved papers, check whether
        # their authors are connected through the co-authorship network --
        # this is the "Dr. X co-authored with Dr. Y" narrative thread.
        author_map = {
            pid: self.graph_store.authors_of(pid) if hasattr(self.graph_store, "authors_of") else []
            for pid in retrieved_ids
        }
        seen_pairs: set[tuple[str, str]] = set()
        for i, pid_a in enumerate(retrieved_ids):
            for pid_b in retrieved_ids[i + 1:]:
                for author_a in author_map.get(pid_a, []):
                    for author_b in author_map.get(pid_b, []):
                        if author_a == author_b:
                            continue
                        pair_key = tuple(sorted((author_a, author_b)))
                        if pair_key in seen_pairs:
                            continue
                        seen_pairs.add(pair_key)
                        path = self.graph_store.shortest_coauthorship_path(author_a, author_b)
                        if path and len(path) > 1:
                            coauthorship_links.append((author_a, author_b, path))

        return citation_paths, shared_methodology, coauthorship_links

    # ------------------------------------------------------------------ #
    # Stage 3: context assembly -> prompt
    # ------------------------------------------------------------------ #
    def build_context(self, query: str, top_k: int = 5) -> GraphRAGContext:
        retrieved = self._retrieve(query, top_k=top_k)
        citation_paths, shared_methodology, coauthorship_links = self._expand_graph_context(retrieved)
        return GraphRAGContext(
            query=query,
            retrieved_papers=retrieved,
            citation_paths=citation_paths,
            shared_methodology=shared_methodology,
            coauthorship_links=coauthorship_links,
        )

    def _resolve_author_label(self, aid: str) -> str:
        if hasattr(self.graph_store, "g") and aid in self.graph_store.g.nodes:
            data = self.graph_store.g.nodes[aid]
            name = data.get("name", aid)
            inst = data.get("institution", "")
            return f"{name} ({inst})" if inst else name
        return aid

    def _resolve_paper_label(self, pid: str) -> str:
        if hasattr(self.graph_store, "get_paper"):
            p = self.graph_store.get_paper(pid)
            if p:
                return f"[{pid}] \"{p.get('title', '')}\""
        return f"[{pid}]"

    def _format_context_as_prompt(self, ctx: GraphRAGContext) -> str:
        lines: list[str] = [f'Research query: "{ctx.query}"', ""]

        lines.append("## Retrieved papers (semantic vector search)")
        for p in ctx.retrieved_papers:
            author_names = []
            if hasattr(self.graph_store, "authors_of"):
                for aid in self.graph_store.authors_of(p["id"]):
                    author_names.append(self._resolve_author_label(aid))
            author_str = f" | Authors: {', '.join(author_names)}" if author_names else ""
            lines.append(
                f"- [{p['id']}] \"{p['title']}\" ({p.get('year', '?')}, "
                f"{p.get('field', 'Unknown field')}{author_str}) — {p['abstract']}"
            )

        lines.append("\n## Citation chains (graph traversal, up to 2 hops)")
        for paper_id, chains in ctx.citation_paths.items():
            if not chains:
                continue
            for chain in chains[:3]:
                resolved_chain = [self._resolve_paper_label(pid) for pid in chain]
                lines.append(f"- {' cites -> '.join(resolved_chain)}")

        lines.append("\n## Shared-methodology neighbors")
        for paper_id, related in ctx.shared_methodology.items():
            if related:
                resolved_related = [self._resolve_paper_label(pid) for pid in related]
                lines.append(
                    f"- {self._resolve_paper_label(paper_id)} shares its methodology with: {', '.join(resolved_related)}"
                )

        lines.append("\n## Co-authorship links between retrieved papers' authors")
        for author_a, author_b, path in ctx.coauthorship_links:
            resolved_path = [self._resolve_author_label(aid) for aid in path]
            lines.append(
                f"- {self._resolve_author_label(author_a)} <-> {self._resolve_author_label(author_b)} "
                f"via path: {' -> '.join(resolved_path)}"
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Stage 4: LLM synthesis
    # ------------------------------------------------------------------ #
    SYSTEM_PROMPT = (
        "You are ScholarRAG, an academic research assistant. You are given "
        "a research query plus structured context extracted from a citation "
        "and co-authorship knowledge graph combined with semantic search "
        "over paper abstracts. Write a concise, plain-English literature "
        "review (3-5 short paragraphs) that: (1) summarizes what the "
        "retrieved papers collectively say about the query, (2) explicitly "
        "explains citation relationships between papers using the provided "
        "graph paths (e.g. 'Paper A builds on Paper B by...'), and (3) "
        "surfaces potential collaboration opportunities based on the "
        "co-authorship links and shared methodologies. Do not invent "
        "papers, authors, or facts beyond what is given in the context."
    )

    def run(self, query: str, top_k: int = 5) -> dict[str, Any]:
        """
        End-to-end pipeline: retrieve -> expand -> synthesize.
        Returns both the LLM's natural-language output AND the structured
        context (so app.py can render the graph visualization from the same
        source of truth used to write the summary).
        """
        ctx = self.build_context(query, top_k=top_k)
        prompt = self._format_context_as_prompt(ctx)
        summary = self.llm.generate(prompt, system_prompt=self.SYSTEM_PROMPT)
        return {
            "query": query,
            "summary": summary,
            "context": ctx,
            "prompt_used": prompt,  # exposed for transparency/debugging in the UI
        }


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    from graph_store import get_default_graph_store
    from vector_store import build_default_vector_store

    gs = get_default_graph_store()
    vs = build_default_vector_store()

    # Falls back to offline template mode automatically if no API key is set.
    pipeline = GraphRAGPipeline(gs, vs, llm_provider=LLMProvider(provider="groq"))
    result = pipeline.run("graph neural networks for scientific recommendation", top_k=4)

    print("=" * 70)
    print("PROMPT SENT TO LLM:\n")
    print(result["prompt_used"])
    print("=" * 70)
    print("\nLLM SUMMARY:\n")
    print(result["summary"])

