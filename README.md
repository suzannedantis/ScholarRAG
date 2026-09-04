# ScholarRAG: Academic Citation & Co-Authorship Navigator

A hybrid **GraphRAG** system connecting researchers to relevant literature,
missing citations, and potential collaborators — combining citation-graph
topology, collaborative filtering, and semantic vector search, entirely on
free/open/embedded infrastructure.

## Architecture

```
data_generator.py  -->  data/*.json  (papers, authors, reading logs, ...)
        |
        v
graph_store.py   -->  NetworkX (default) or Kùzu embedded graph DB
vector_store.py  -->  ChromaDB (embedded) + sentence-transformers (local, free)
        |
        v
recommender.py   -->  Implicit ALS + graph centrality + vector similarity (blended)
graph_rag.py     -->  vector retrieval + graph expansion -> Groq/Gemini LLM synthesis
        |
        v
app.py           -->  Streamlit UI (recommendation cards, pyvis graph, LLM summary)
```

## Setup

```bash
pip install -r requirements.txt

# Optional (only if you enable the LLM synthesis step in graph_rag.py):
export GROQ_API_KEY="..."      # https://console.groq.com  (free tier)
# or
export GEMINI_API_KEY="..."    # https://aistudio.google.com (free tier)
```

## Run

```bash
# 1. Generate the mock dataset (30 papers, 18 authors, reading logs, ...)
python data_generator.py

# 2. (Optional) sanity-check each backend module standalone
python graph_store.py
python vector_store.py
python recommender.py
python graph_rag.py

# 3. Launch the app
streamlit run app.py
```

If no `GROQ_API_KEY` / `GEMINI_API_KEY` is set, `graph_rag.py` automatically
falls back to an offline template response so the rest of the app (graph
traversal, recommendations, visualization) remains fully demoable without
any API key.

## Notes on design choices

- **NetworkX vs Kùzu**: `graph_store.py` defines `BaseGraphStore` as a shared
  interface implemented by both `NetworkXGraphStore` (default, zero-setup,
  in-memory) and `KuzuGraphStore` (embedded on-disk, Cypher-like queries,
  install via `pip install kuzu`). Swap engines by changing one line in
  `get_default_graph_store()`.
- **Citation DAG is acyclic by construction**: `data_generator.py` only lets
  papers cite strictly earlier papers, which is what makes citation-depth
  and multi-hop traversal well-defined (no infinite cycles).
- **Cold start handling**: `recommender.py` redistributes blend weights
  across active signals only, so a new user with no reading history (or a
  query-only search) still gets a sensible ranking from graph + semantic
  signals alone.
- **GraphRAG context is a first-class object** (`GraphRAGContext`), not just
  a prompt string — `app.py` renders it both as the LLM prompt and directly
  in the "show retrieved context" debug panel, so what the LLM saw is always
  inspectable.
