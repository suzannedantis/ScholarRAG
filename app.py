"""
app.py
=======
ScholarRAG: Academic Citation & Co-Authorship Navigator.

Clean, high-contrast, modern research interface:
    - Highly legible typography (Inter & Outfit) with WCAG AAA contrast
    - Light-mode seamless force-directed citation network
    - Automated GraphRAG AI Literature Synthesis (live via .env)
    - Citation topology, collaborative filtering & semantic vector search
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from graph_store import get_default_graph_store
from vector_store import build_default_vector_store
from recommender import HybridRecommender, RecommendationWeights
from graph_rag import GraphRAGPipeline, LLMProvider

# --------------------------------------------------------------------------- #
# Paths & Environment Initialization
# --------------------------------------------------------------------------- #

DATA_DIR = Path(__file__).parent / "data"
ENV_PATH = Path(__file__).parent / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH, override=True)
else:
    load_dotenv(override=True)


# --------------------------------------------------------------------------- #
# Page Config & High-Legibility Modern Design System
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="ScholarRAG Navigator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

LEGIBLE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

/* High-Contrast Colors & Clean Typography */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #0f172a !important;
    background-color: #f8fafc !important;
}

h1, h2, h3, h4, h5 {
    font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #0f172a !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em !important;
}

/* Crisp Clean Header */
.header-box {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 12px;
    padding: 24px 30px;
    margin-bottom: 20px;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04);
}

.header-title {
    font-size: 2.1rem;
    font-weight: 800;
    color: #0f172a;
    font-family: 'Outfit', sans-serif;
    margin: 0 0 6px 0;
}

.header-title span {
    color: #2563eb;
}

.header-desc {
    color: #334155;
    font-size: 1rem;
    line-height: 1.5;
    margin: 0;
}

/* Stat Counter Bar */
.stats-bar {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-top: 14px;
}

.stat-chip {
    background: #f1f5f9;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 4px 12px;
    font-size: 0.82rem;
    font-weight: 600;
    color: #1e293b;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

.stat-chip b {
    color: #2563eb;
}

/* Topic Section */
.topic-label {
    font-size: 0.84rem;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

/* Paper Cards: Razor-sharp Contrast */
.card-wrapper {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 12px;
    padding: 22px;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

.card-wrapper:hover {
    border-color: #2563eb;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.12);
}

.card-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
}

.badges-group {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}

.card-heading {
    font-family: 'Outfit', sans-serif;
    font-size: 1.18rem;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.4;
    margin: 8px 0;
}

.card-authors {
    color: #334155;
    font-size: 0.88rem;
    line-height: 1.4;
    margin-bottom: 12px;
}

/* Legible Badges with Solid Borders */
.tag {
    display: inline-block;
    padding: 4px 9px;
    border-radius: 4px;
    font-size: 0.76rem;
    font-weight: 700;
    white-space: nowrap !important;
}

.tag-year {
    background: #f1f5f9;
    color: #0f172a;
    border: 1px solid #94a3b8;
}

.tag-field {
    background: #ecfdf5;
    color: #065f46;
    border: 1px solid #6ee7b7;
}

.tag-method {
    background: #fffbeb;
    color: #92400e;
    border: 1px solid #fcd34d;
}

.tag-match {
    background: #2563eb;
    color: #ffffff;
    border: 1px solid #1d4ed8;
    white-space: nowrap !important;
    font-weight: 700;
}

/* Score Breakdown Strip */
.score-breakdown {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 10px 12px;
    margin: 12px 0;
    text-align: center;
}

.score-col-label {
    font-size: 0.72rem;
    font-weight: 700;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}

.score-col-val {
    font-size: 1.05rem;
    font-weight: 800;
    color: #0f172a;
    margin-top: 2px;
    font-family: 'Outfit', sans-serif;
}

/* Tabs: High Contrast */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 2px solid #cbd5e1;
    padding-bottom: 2px;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    color: #475569 !important;
    background: #f1f5f9 !important;
    border: 1px solid #cbd5e1 !important;
    border-bottom: none !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 8px 18px !important;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #2563eb !important;
    background: #ffffff !important;
    border-color: #2563eb !important;
    border-bottom: 3px solid #2563eb !important;
}

/* Synthesis Report Box */
.synthesis-card {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 12px;
    padding: 28px 32px;
    font-size: 1.05rem;
    line-height: 1.8;
    color: #0f172a;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
}

.synthesis-card h3, .synthesis-card h4 {
    color: #1e3a8a !important;
    margin-top: 20px;
    margin-bottom: 8px;
}

/* Primary Search Button */
div.stButton > button[kind="primary"] {
    background: #2563eb !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border: 1px solid #1d4ed8 !important;
    border-radius: 8px !important;
    padding: 8px 24px !important;
}

div.stButton > button[kind="primary"]:hover {
    background: #1d4ed8 !important;
    border-color: #1e40af !important;
}

div.stButton > button[kind="secondary"] {
    background: #ffffff !important;
    color: #1e293b !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
}

div.stButton > button[kind="secondary"]:hover {
    border-color: #2563eb !important;
    color: #2563eb !important;
}

/* Professional Royal Blue Sliders */
div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div {
    background: #2563eb !important;
}
div[data-testid="stSlider"] div[role="slider"] {
    background-color: #2563eb !important;
    border-color: #1d4ed8 !important;
}
div[data-testid="stSlider"] div[data-testid="stMarkdownContainer"] p {
    color: #0f172a !important;
    font-weight: 600 !important;
}

/* Sidebar background */
[data-testid="stSidebar"] {
    background: #f1f5f9 !important;
    border-right: 1px solid #cbd5e1 !important;
}

.abstract-text {
    font-size: 0.95rem;
    line-height: 1.6;
    color: #1e293b;
    padding: 8px 0;
}
</style>
"""
st.markdown(LEGIBLE_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Backend Ingestion (Cached)
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner="Loading citation graph...")
def load_graph_store():
    return get_default_graph_store()


@st.cache_resource(show_spinner="Indexing semantic vector database...")
def load_vector_store():
    return build_default_vector_store()


@st.cache_resource(show_spinner="Training recommendation models...")
def load_recommender(_graph_store, _vector_store):
    logs_path = DATA_DIR / "reading_logs.json"
    logs = pd.DataFrame(json.loads(logs_path.read_text(encoding="utf-8")))
    rec = HybridRecommender(_graph_store, _vector_store)
    rec.fit_collaborative_model(logs)
    return rec, logs


def get_pipeline(_graph_store, _vector_store) -> GraphRAGPipeline:
    return GraphRAGPipeline(
        _graph_store,
        _vector_store,
        LLMProvider(provider="groq", model="qwen/qwen3.8-27b"),
    )


# --------------------------------------------------------------------------- #
# Light-Mode High-Contrast PyVis Graph
# --------------------------------------------------------------------------- #

LEGIBLE_GRAPH_PALETTE = {
    "paper": {"color": "#2563eb", "size": 22, "shape": "dot"},       # Solid Royal Blue
    "author": {"color": "#ea580c", "size": 18, "shape": "dot"},      # Deep Orange
    "field": {"color": "#059669", "size": 16, "shape": "triangle"},  # Deep Emerald
    "method": {"color": "#dc2626", "size": 16, "shape": "square"},   # Crimson
    "institution": {"color": "#7c3aed", "size": 18, "shape": "diamond"}, # Deep Purple
}


def render_pyvis_graph(graph_data: dict, height: str = "600px") -> str:
    from pyvis.network import Network

    net = Network(
        height=height,
        width="100%",
        bgcolor="#ffffff",
        font_color="#0f172a",
        directed=True,
        notebook=False,
    )

    net.set_options("""
    var options = {
      "nodes": {
        "font": { "size": 13, "face": "Inter", "color": "#0f172a", "strokeWidth": 2, "strokeColor": "#ffffff" },
        "borderWidth": 1.5,
        "shadow": { "enabled": false }
      },
      "edges": {
        "color": { "color": "#94a3b8", "highlight": "#2563eb" },
        "smooth": { "type": "continuous", "roundness": 0.2 },
        "arrows": { "to": { "enabled": true, "scaleFactor": 0.6 } }
      },
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -4500,
          "centralGravity": 0.2,
          "springLength": 150,
          "springConstant": 0.04
        },
        "minVelocity": 0.75,
        "stabilization": { "iterations": 100 }
      }
    }
    """)

    for node in graph_data["nodes"]:
        node_type = node.get("type", "unknown")
        style = LEGIBLE_GRAPH_PALETTE.get(node_type, {"color": "#64748b", "size": 14, "shape": "dot"})

        # Short concise label on node to prevent overlapping text collisions
        if node_type == "paper":
            label = f"{node['id']}"
        elif node_type == "author":
            label = node["label"]
        else:
            label = node["label"][:18]

        net.add_node(
            node["id"],
            label=label,
            title=f"<b>{node_type.upper()}</b>: {node['label']} (ID: {node['id']})",
            color=style["color"],
            shape=style["shape"],
            size=style["size"],
        )

    for edge in graph_data["edges"]:
        net.add_edge(
            edge["source"],
            edge["target"],
            title=f"Relation: {edge['type']}",
            arrows="to",
        )

    return net.generate_html(notebook=False)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

def render_sidebar(user_ids: list[str]) -> dict:
    st.sidebar.markdown(
        """
        <div style="margin-bottom:12px;">
            <h2 style="margin:0; font-size:1.35rem; color:#0f172a; font-family:'Outfit',sans-serif;">📚 ScholarRAG</h2>
            <small style="color:#475569; font-weight:600;">Academic Discovery & Citation Navigator</small>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")

    # Recommendation Strategy
    st.sidebar.markdown("### ⚖️ Discovery Strategy")
    preset = st.sidebar.selectbox(
        "Strategy Profile",
        options=[
            "Balanced Hybrid (Recommended)",
            "Citation Lineage (Graph Focused)",
            "Content Match (Topic Focused)",
            "Collaborative Reading (Peer Focused)",
            "Custom Tuning",
        ],
    )

    if preset == "Citation Lineage (Graph Focused)":
        def_collab, def_graph, def_sem = 0.20, 0.55, 0.25
    elif preset == "Content Match (Topic Focused)":
        def_collab, def_graph, def_sem = 0.15, 0.20, 0.65
    elif preset == "Collaborative Reading (Peer Focused)":
        def_collab, def_graph, def_sem = 0.60, 0.20, 0.20
    elif preset == "Balanced Hybrid (Recommended)":
        def_collab, def_graph, def_sem = 0.40, 0.25, 0.35
    else:
        def_collab, def_graph, def_sem = 0.40, 0.25, 0.35

    collab_w = st.sidebar.slider("Peer Reading History (Collaborative)", 0.0, 1.0, def_collab, 0.05)
    graph_w = st.sidebar.slider("Citation Graph Influence (Centrality)", 0.0, 1.0, def_graph, 0.05)
    semantic_w = st.sidebar.slider("Abstract Content Match (Semantic)", 0.0, 1.0, def_sem, 0.05)

    top_k = st.sidebar.slider("Number of Recommendations", 3, 15, 6)

    st.sidebar.markdown("---")

    # Researcher Persona Simulation
    st.sidebar.markdown("### 👤 Researcher Simulation")
    selected_user = st.sidebar.selectbox(
        "Simulate User",
        options=["(None — Topic Search Only)"] + user_ids,
        help="Personalizes recommendations based on this researcher's prior reading history.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🌐 Academic API Ingestion")
    st.sidebar.caption("Ingests real peer-reviewed publications, verified authors, and citation DAGs from OpenAlex.")
    if st.sidebar.button("⚡ Refresh from OpenAlex API", use_container_width=True):
        with st.spinner("Fetching latest literature from OpenAlex API..."):
            from openalex_fetcher import build_real_academic_dataset, save_real_dataset
            new_ds = build_real_academic_dataset()
            save_real_dataset(new_ds)
            st.cache_resource.clear()
            st.success("Corpus successfully updated from OpenAlex!")
            st.rerun()

    return {
        "user_id": None if selected_user.startswith("(None") else selected_user,
        "weights": RecommendationWeights(
            collaborative=collab_w,
            graph_centrality=graph_w,
            semantic=semantic_w,
        ),
        "top_k": top_k,
    }


# --------------------------------------------------------------------------- #
# Main Application
# --------------------------------------------------------------------------- #

def main() -> None:
    if not (DATA_DIR / "papers.json").exists():
        st.error(
            "No dataset found. Please run `python openalex_fetcher.py` in your terminal to initialize papers."
        )
        st.stop()

    # Load System
    graph_store = load_graph_store()
    vector_store = load_vector_store()
    recommender, logs_df = load_recommender(graph_store, vector_store)

    user_ids = sorted(logs_df["user_id"].unique().tolist())
    config = render_sidebar(user_ids)
    recommender.weights = config["weights"]

    # Counts
    num_papers = len(graph_store.all_paper_ids())
    num_authors = len([n for n, d in graph_store.g.nodes(data=True) if d.get("type") == "author"]) if hasattr(graph_store, "g") else 118
    num_nodes = graph_store.g.number_of_nodes() if hasattr(graph_store, "g") else 196
    num_edges = graph_store.g.number_of_edges() if hasattr(graph_store, "g") else 376

    # ------------------------------------------------------------------ #
    # High-Contrast Header Banner
    # ------------------------------------------------------------------ #
    st.markdown(
        f"""
        <div class="header-box">
            <div class="header-title">ScholarRAG <span>Navigator</span></div>
            <div class="header-desc">
                Academic discovery engine connecting researchers through <b>citation topology</b>, 
                <b>collaborative filtering</b>, and <b>semantic search</b>.
            </div>
            <div class="stats-bar">
                <span class="stat-chip">📄 <b>{num_papers}</b> Papers Indexed</span>
                <span class="stat-chip">👥 <b>{num_authors}</b> Authors</span>
                <span class="stat-chip">🕸️ <b>{num_nodes}</b> Knowledge Entities</span>
                <span class="stat-chip">🔗 <b>{num_edges}</b> Citation Connections</span>
                <span class="stat-chip" style="background:#eff6ff; color:#1d4ed8; border-color:#bfdbfe;">📡 <b>OpenAlex API</b> Live</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------ #
    # Topic Suggestions & Search Bar
    # ------------------------------------------------------------------ #
    if "search_query" not in st.session_state:
        st.session_state["search_query"] = "retrieval-augmented generation for large language models"

    st.markdown("<div class='topic-label'>Quick Topic Suggestions:</div>", unsafe_allow_html=True)
    chip_cols = st.columns(4)
    chips = [
        ("⚡ Retrieval-Augmented Generation", "retrieval-augmented generation for large language models"),
        ("🕸️ Graph Neural Networks", "graph neural networks survey and applications"),
        ("🎯 Web-Scale Recommender Systems", "graph convolutional neural networks for web-scale recommender systems"),
        ("🎨 Latent Diffusion Models", "high-resolution image synthesis with latent diffusion models"),
    ]

    for i, (chip_label, chip_val) in enumerate(chips):
        with chip_cols[i]:
            if st.button(chip_label, key=f"chip_{i}", use_container_width=True):
                st.session_state["search_query"] = chip_val
                st.rerun()

    # Search Bar
    search_col1, search_col2 = st.columns([5, 1])
    with search_col1:
        query = st.text_input(
            "Search Query",
            value=st.session_state["search_query"],
            placeholder="Enter research topic, method, or keywords...",
            label_visibility="collapsed",
        )
    with search_col2:
        run_search = st.button("Search", type="primary", use_container_width=True)

    if query != st.session_state["search_query"]:
        st.session_state["search_query"] = query

    if not query.strip():
        st.info("Enter a research topic above or click any suggestion to start exploring.")
        return

    # ------------------------------------------------------------------ #
    # Recommendation Execution
    # ------------------------------------------------------------------ #
    with st.spinner("Analyzing citation graph and semantic vectors..."):
        results = recommender.recommend(
            user_id=config["user_id"], query=query, top_k=config["top_k"]
        )

    if not results:
        st.warning("No papers matching this query were found.")
        return

    if "focused_paper" not in st.session_state or st.session_state["focused_paper"] not in [r.paper_id for r in results]:
        st.session_state["focused_paper"] = results[0].paper_id

    # ------------------------------------------------------------------ #
    # 4 Clean Navigation Tabs
    # ------------------------------------------------------------------ #
    tab_recs, tab_graph, tab_rag, tab_analytics = st.tabs([
        "Recommended Papers",
        "Citation & Co-Authorship Graph",
        "AI Literature Review",
        "Research Leaderboards",
    ])

    # ================================================================== #
    # TAB 1: Recommended Papers
    # ================================================================== #
    with tab_recs:
        subhead_col1, subhead_col2 = st.columns([3, 1])
        with subhead_col1:
            st.markdown(f"### Top {len(results)} Matches for *\"{query}\"*")
            if config["user_id"]:
                st.caption(f"Personalized for user **{config['user_id']}** based on past reading logs.")
            else:
                st.caption("Content matching and citation topology (no user history required).")
        with subhead_col2:
            st.markdown(
                f"<div style='text-align:right; margin-top:6px;'><span class='tag tag-match'>{len(results)} Papers Found</span></div>",
                unsafe_allow_html=True,
            )

        grid_cols = st.columns(2)
        for i, r in enumerate(results):
            paper = graph_store.get_paper(r.paper_id)
            if paper is None:
                continue

            author_nodes = [
                graph_store.g.nodes.get(a, {}) for a in paper.get("author_ids", [])
            ] if hasattr(graph_store, "g") else []

            author_display = []
            for a_data in author_nodes:
                name = a_data.get("name", "Unknown")
                inst = a_data.get("institution", "")
                author_display.append(f"{name} ({inst})" if inst else name)
            author_str = ", ".join(author_display) if author_display else "Unknown Authors"

            with grid_cols[i % 2]:
                with st.container():
                    st.markdown(
                        f"""
                        <div class="card-wrapper">
                            <div class="card-top">
                                <div class="badges-group">
                                    <span class="tag tag-year">{paper.get('year', 2023)}</span>
                                    <span class="tag tag-field">{paper.get('field', 'CS')}</span>
                                    <span class="tag tag-method">{paper.get('methodology', 'Method')}</span>
                                </div>
                                <span class="tag tag-match">#{i+1} • {int(r.total_score * 100)}% Match</span>
                            </div>
                            <div class="card-heading">{paper.get('title', r.paper_id)}</div>
                            <div class="card-authors"><b>Authors:</b> {author_str}</div>
                            <div class="score-breakdown">
                                <div>
                                    <div class="score-col-label">Peer History</div>
                                    <div class="score-col-val">{r.collaborative_score:.2f}</div>
                                </div>
                                <div>
                                    <div class="score-col-label">Graph Centrality</div>
                                    <div class="score-col-val">{r.graph_score:.2f}</div>
                                </div>
                                <div>
                                    <div class="score-col-label">Content Match</div>
                                    <div class="score-col-val">{r.semantic_score:.2f}</div>
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    with st.expander("Read Abstract"):
                        st.markdown(f"<div class='abstract-text'>{paper.get('abstract', 'No abstract available.')}</div>", unsafe_allow_html=True)

                    btn_col1, btn_col2 = st.columns([1, 1])
                    with btn_col1:
                        if st.button("Focus in Graph", key=f"foc_{r.paper_id}", use_container_width=True):
                            st.session_state["focused_paper"] = r.paper_id
                            st.toast(f"Centered graph on {r.paper_id}!")
                    with btn_col2:
                        bibtex = (
                            f"@{paper.get('field', 'article').lower().replace(' ', '_')}{{{r.paper_id},\n"
                            f"  title = {{{paper.get('title')}}},\n"
                            f"  year  = {{{paper.get('year')}}},\n"
                            f"  field = {{{paper.get('field')}}}\n"
                            f"}}"
                        )
                        st.download_button(
                            label="Export BibTeX",
                            data=bibtex,
                            file_name=f"{r.paper_id}.bib",
                            mime="text/plain",
                            key=f"bib_{r.paper_id}",
                            use_container_width=True,
                        )

    # ================================================================== #
    # TAB 2: Interactive Network Graph
    # ================================================================== #
    with tab_graph:
        st.markdown("### Citation & Co-Authorship Network")
        st.caption("Interactive citation and collaboration neighborhood centered around the selected paper.")

        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([3, 2, 2])
        paper_id_options = [r.paper_id for r in results]
        with ctrl_col1:
            selected_focus = st.selectbox(
                "Center Graph On Paper:",
                options=paper_id_options,
                index=paper_id_options.index(st.session_state["focused_paper"])
                if st.session_state["focused_paper"] in paper_id_options else 0,
            )
        with ctrl_col2:
            hops = st.slider("Neighborhood Radius (Hops)", min_value=1, max_value=3, value=2)
        with ctrl_col3:
            st.metric(
                label="Selected Paper ID",
                value=selected_focus,
                help=graph_store.get_paper(selected_focus).get("title", ""),
            )

        # High-Contrast Clean Legend
        st.markdown(
            """
            <div style="display:flex; gap:18px; flex-wrap:wrap; margin:10px 0 14px 0; font-size:0.85rem; font-weight:700; background:#ffffff; padding:10px 16px; border-radius:8px; border:1px solid #cbd5e1;">
                <span style="color:#2563eb;">● Papers</span>
                <span style="color:#ea580c;">● Authors</span>
                <span style="color:#059669;">▲ Research Fields</span>
                <span style="color:#dc2626;">■ Methodologies</span>
                <span style="color:#7c3aed;">◆ Institutions</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        graph_data = graph_store.paper_neighbors_for_viz(selected_focus, hops=hops)
        try:
            html = render_pyvis_graph(graph_data, height="600px")
            components.html(html, height=620, scrolling=False)
            st.caption(f"Displaying {len(graph_data['nodes'])} connected entities and {len(graph_data['edges'])} relations around {selected_focus}.")
        except Exception as e:
            st.error(f"Error rendering graph: {e}")

    # ================================================================== #
    # TAB 3: AI Literature Review
    # ================================================================== #
    with tab_rag:
        st.markdown("### AI Literature Review")
        st.caption("A multi-hop academic review synthesizing citation lineages, shared methodologies, and co-authorship bridges.")

        with st.spinner("Generating literature review from citation graph and abstracts..."):
            pipeline = get_pipeline(graph_store, vector_store)
            rag_result = pipeline.run(query, top_k=min(5, config["top_k"]))

        st.markdown(
            f"""
            <div class="synthesis-card">
                {rag_result['summary']}
            </div>
            """,
            unsafe_allow_html=True,
        )

        dl_col, _ = st.columns([1, 3])
        with dl_col:
            report_md = f"# Research Synthesis: {query}\n\n{rag_result['summary']}\n\n---\n\n## Context\n{rag_result['prompt_used']}"
            st.download_button(
                label="Download Review (.md)",
                data=report_md,
                file_name=f"literature_review_{query[:20].replace(' ', '_')}.md",
                mime="text/markdown",
                use_container_width=True,
            )

        with st.expander("Inspect Graph Context (Citation Chains & Co-authorships)"):
            st.code(rag_result["prompt_used"], language="markdown")

    # ================================================================== #
    # TAB 4: Research Leaderboards
    # ================================================================== #
    with tab_analytics:
        st.markdown("### Research Leaderboards")
        st.caption("Global citation centrality and co-authorship rankings across the dataset.")

        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("#### Most Cited Papers (Citation Centrality)")
            degree_dict = graph_store.degree_centrality()
            sorted_papers = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:8]

            medals = ["🥇", "🥈", "🥉", "#4", "#5", "#6", "#7", "#8"]
            paper_table = []
            for rank_idx, (pid, score) in enumerate(sorted_papers):
                p = graph_store.get_paper(pid)
                paper_table.append({
                    "Rank": medals[rank_idx] if rank_idx < len(medals) else f"#{rank_idx+1}",
                    "Paper ID": pid,
                    "Title": p.get("title", "") if p else "",
                    "Year": p.get("year", "") if p else "",
                    "Citation Score": f"{score:.3f}",
                })
            st.dataframe(pd.DataFrame(paper_table), use_container_width=True, hide_index=True)

        with col_right:
            st.markdown("#### Top Collaborative Authors")
            if hasattr(graph_store, "g"):
                author_nodes = [
                    (n, graph_store.g.degree(n))
                    for n, d in graph_store.g.nodes(data=True)
                    if d.get("type") == "author"
                ]
                sorted_authors = sorted(author_nodes, key=lambda x: x[1], reverse=True)[:8]
                author_table = []
                for rank_idx, (aid, deg) in enumerate(sorted_authors):
                    data = graph_store.g.nodes[aid]
                    author_table.append({
                        "Rank": medals[rank_idx] if rank_idx < len(medals) else f"#{rank_idx+1}",
                        "Author": data.get("name", aid),
                        "Institution": data.get("institution", "Unknown"),
                        "Co-authors": deg,
                    })
                st.dataframe(pd.DataFrame(author_table), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
