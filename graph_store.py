"""
graph_store.py
===============
Ingests ScholarRAG's mock dataset into a property graph and exposes graph
traversal primitives used by both the recommender and the GraphRAG pipeline:

    - shortest co-authorship path between two authors (BFS over an
      author-author projection built from shared AUTHORED edges)
    - citation depth / multi-hop citation chains (BFS/DFS over CITES edges)
    - shared methodology / field lookups
    - degree & centrality metrics (used as a graph-based recommender signal)

Two engines are supported behind one interface, `BaseGraphStore`:

    - `NetworkXGraphStore`  -- pure-Python, in-memory, zero setup. Default.
    - `KuzuGraphStore`      -- embedded on-disk graph DB (Cypher-like query
                               language), better suited to larger graphs.
                               Requires `pip install kuzu`.

Nodes:  Paper, Author, FieldOfStudy, Methodology, Institution
Edges:  AUTHORED (Author->Paper), CITES (Paper->Paper),
        USES_METHOD (Paper->Methodology), BELONGS_TO (Paper->FieldOfStudy),
        AFFILIATED_WITH (Author->Institution)
"""

from __future__ import annotations

import json
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import networkx as nx

DATA_DIR = Path(__file__).parent / "data"


class BaseGraphStore(ABC):
    """Common interface so recommender.py / graph_rag.py never care which
    physical graph engine is running underneath."""

    @abstractmethod
    def load(self, data_dir: Path = DATA_DIR) -> None: ...

    @abstractmethod
    def shortest_coauthorship_path(self, author_a: str, author_b: str) -> Optional[list[str]]: ...

    @abstractmethod
    def citation_chain(self, paper_id: str, max_depth: int = 3) -> list[list[str]]: ...

    @abstractmethod
    def shared_methodology_papers(self, paper_id: str) -> list[str]: ...

    @abstractmethod
    def paper_neighbors_for_viz(self, paper_id: str, hops: int = 2) -> dict[str, Any]: ...

    @abstractmethod
    def degree_centrality(self) -> dict[str, float]: ...


# --------------------------------------------------------------------------- #
# NetworkX implementation (default, embedded, no external services required)
# --------------------------------------------------------------------------- #

class NetworkXGraphStore(BaseGraphStore):
    """
    A single `networkx.MultiDiGraph` holds every node type, disambiguated by
    a `type` node attribute (`paper`, `author`, `field`, `method`, `institution`)
    and every edge type via an `edge_type` edge attribute. This keeps the
    implementation simple while still letting us answer heterogeneous graph
    queries (co-authorship, citation depth, shared methodology).
    """

    def __init__(self) -> None:
        self.g = nx.MultiDiGraph()
        self._loaded = False

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #
    def load(self, data_dir: Path = DATA_DIR) -> None:
        try:
            authors = json.loads((data_dir / "authors.json").read_text(encoding="utf-8"))
            papers = json.loads((data_dir / "papers.json").read_text(encoding="utf-8"))
            fields = json.loads((data_dir / "fields_of_study.json").read_text(encoding="utf-8"))
            methods = json.loads((data_dir / "methodologies.json").read_text(encoding="utf-8"))
            institutions = json.loads((data_dir / "institutions.json").read_text(encoding="utf-8"))
        except FileNotFoundError as e:
            raise FileNotFoundError(
                "Academic dataset not found. Run `python openalex_fetcher.py` first."
            ) from e

        # --- Nodes ---
        for f in fields:
            self.g.add_node(f"field::{f}", type="field", name=f)
        for m in methods:
            self.g.add_node(f"method::{m}", type="method", name=m)
        for inst in institutions:
            self.g.add_node(f"inst::{inst}", type="institution", name=inst)
        for a in authors:
            self.g.add_node(
                a["id"],
                type="author",
                name=a["name"],
                institution=a.get("institution", "Unknown"),
            )
            self.g.add_edge(
                a["id"], f"inst::{a['institution']}", edge_type="AFFILIATED_WITH"
            )
        for p in papers:
            self.g.add_node(
                p["id"],
                type="paper",
                title=p["title"],
                abstract=p["abstract"],
                year=p["year"],
                field=p["field"],
                methodology=p["methodology"],
            )
            self.g.add_edge(p["id"], f"field::{p['field']}", edge_type="BELONGS_TO")
            self.g.add_edge(
                p["id"], f"method::{p['methodology']}", edge_type="USES_METHOD"
            )
            for author_id in p["author_ids"]:
                self.g.add_edge(author_id, p["id"], edge_type="AUTHORED")
            for cited_id in p["citation_ids"]:
                # p CITES cited_id  (p -> cited_id, direction = "cites")
                self.g.add_edge(p["id"], cited_id, edge_type="CITES")

        self._loaded = True
        n_nodes, n_edges = self.g.number_of_nodes(), self.g.number_of_edges()
        print(f"[graph_store:NetworkX] Loaded graph with {n_nodes} nodes, {n_edges} edges.")

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("Graph not loaded. Call .load() first.")

    # ------------------------------------------------------------------ #
    # Co-authorship traversal
    # ------------------------------------------------------------------ #
    def _coauthor_projection(self) -> nx.Graph:
        """
        Builds an undirected author-author graph where an edge exists if two
        authors share at least one AUTHORED edge to the same paper. This is
        the classic bipartite-to-unipartite projection used for
        co-authorship networks.
        """
        coauthor_graph = nx.Graph()
        paper_nodes = [n for n, d in self.g.nodes(data=True) if d.get("type") == "paper"]
        for paper_id in paper_nodes:
            authors = [
                u for u, v, d in self.g.in_edges(paper_id, data=True)
                if d.get("edge_type") == "AUTHORED"
            ]
            for i in range(len(authors)):
                for j in range(i + 1, len(authors)):
                    coauthor_graph.add_edge(authors[i], authors[j], via=paper_id)
        return coauthor_graph

    def shortest_coauthorship_path(self, author_a: str, author_b: str) -> Optional[list[str]]:
        """
        Finds the shortest chain of co-authorship links connecting two
        authors, e.g. ["A001", "A007", "A012"] means A001 co-authored with
        A007, who co-authored with A012. Returns None if disconnected.
        """
        self._require_loaded()
        coauthor_graph = self._coauthor_projection()
        if author_a not in coauthor_graph or author_b not in coauthor_graph:
            return None
        try:
            return nx.shortest_path(coauthor_graph, author_a, author_b)
        except nx.NetworkXNoPath:
            return None

    # ------------------------------------------------------------------ #
    # Citation traversal
    # ------------------------------------------------------------------ #
    def citation_chain(self, paper_id: str, max_depth: int = 3) -> list[list[str]]:
        """
        Multi-hop citation traversal: returns every citation PATH starting
        at `paper_id` up to `max_depth` hops, e.g.
            [["P010", "P004"], ["P010", "P004", "P001"]]
        means P010 cites P004, and P004 in turn cites P001.
        Implemented as bounded-depth DFS over CITES edges.
        """
        self._require_loaded()
        if paper_id not in self.g:
            return []

        paths: list[list[str]] = []

        def dfs(current: str, path: list[str], depth: int) -> None:
            if depth >= max_depth:
                return
            for _, target, data in self.g.out_edges(current, data=True):
                if data.get("edge_type") != "CITES":
                    continue
                new_path = path + [target]
                paths.append(new_path)
                dfs(target, new_path, depth + 1)

        dfs(paper_id, [paper_id], 0)
        return paths

    def citation_depth(self, paper_id: str) -> int:
        """Longest citation chain reachable from this paper (its 'lineage depth')."""
        chains = self.citation_chain(paper_id, max_depth=10)
        return max((len(c) - 1 for c in chains), default=0)

    # ------------------------------------------------------------------ #
    # Shared methodology / field lookups
    # ------------------------------------------------------------------ #
    def shared_methodology_papers(self, paper_id: str) -> list[str]:
        """Other papers using the same primary methodology as `paper_id`."""
        self._require_loaded()
        if paper_id not in self.g:
            return []
        method_nodes = [
            v for _, v, d in self.g.out_edges(paper_id, data=True)
            if d.get("edge_type") == "USES_METHOD"
        ]
        related: set[str] = set()
        for method_node in method_nodes:
            for u, _, d in self.g.in_edges(method_node, data=True):
                if d.get("edge_type") == "USES_METHOD" and u != paper_id:
                    related.add(u)
        return sorted(related)

    def authors_of(self, paper_id: str) -> list[str]:
        self._require_loaded()
        return [
            u for u, _, d in self.g.in_edges(paper_id, data=True)
            if d.get("edge_type") == "AUTHORED"
        ]

    # ------------------------------------------------------------------ #
    # Centrality (used as a graph-based ranking signal in recommender.py)
    # ------------------------------------------------------------------ #
    def degree_centrality(self) -> dict[str, float]:
        """
        In-degree centrality over the CITES sub-graph approximates "influence":
        a paper cited by many others scores higher. This mirrors PageRank-style
        intuition without the extra computational cost for a small demo graph.
        """
        self._require_loaded()
        citation_subgraph = nx.DiGraph(
            (u, v) for u, v, d in self.g.edges(data=True) if d.get("edge_type") == "CITES"
        )
        paper_nodes = [n for n, d in self.g.nodes(data=True) if d.get("type") == "paper"]
        citation_subgraph.add_nodes_from(paper_nodes)
        return nx.in_degree_centrality(citation_subgraph)

    # ------------------------------------------------------------------ #
    # Visualization export (consumed by app.py's pyvis renderer)
    # ------------------------------------------------------------------ #
    def paper_neighbors_for_viz(self, paper_id: str, hops: int = 2) -> dict[str, Any]:
        """
        Returns an ego-graph (nodes + edges) around `paper_id` up to `hops`
        hops, restricted to CITES and AUTHORED edges, in a plain dict format
        that app.py converts into a pyvis Network.
        """
        self._require_loaded()
        if paper_id not in self.g:
            return {"nodes": [], "edges": []}

        undirected = self.g.to_undirected()
        ego = nx.ego_graph(undirected, paper_id, radius=hops)

        nodes = []
        for n, d in ego.nodes(data=True):
            label = d.get("title") or d.get("name") or n
            nodes.append({"id": n, "label": label[:40], "type": d.get("type", "unknown")})

        edges = []
        seen = set()
        for u, v, d in ego.edges(data=True):
            key = (u, v, d.get("edge_type"))
            if key in seen:
                continue
            seen.add(key)
            edges.append({"source": u, "target": v, "type": d.get("edge_type", "related")})

        return {"nodes": nodes, "edges": edges}

    def get_paper(self, paper_id: str) -> Optional[dict[str, Any]]:
        self._require_loaded()
        if paper_id not in self.g:
            return None
        data = dict(self.g.nodes[paper_id])
        data["id"] = paper_id
        data["author_ids"] = self.authors_of(paper_id)
        return data

    def all_paper_ids(self) -> list[str]:
        self._require_loaded()
        return [n for n, d in self.g.nodes(data=True) if d.get("type") == "paper"]


# --------------------------------------------------------------------------- #
# Kùzu implementation (optional, swap-in embedded graph DB)
# --------------------------------------------------------------------------- #

class KuzuGraphStore(BaseGraphStore):
    """
    Same interface as NetworkXGraphStore but backed by Kùzu, an embedded
    (in-process, on-disk) property graph database queried with a Cypher-like
    language. Useful once the graph grows beyond what comfortably fits in
    memory, or when you want persistent storage between runs.

    Requires: pip install kuzu
    """

    def __init__(self, db_path: str = "./scholarrag_kuzu_db") -> None:
        self.db_path = db_path
        self._conn = None
        self._loaded = False

    def _connect(self):
        try:
            import kuzu
        except ImportError as e:
            raise ImportError(
                "Kùzu is not installed. Run `pip install kuzu`, or use "
                "NetworkXGraphStore instead (default, no extra install)."
            ) from e

        # Fresh DB each run for a reproducible demo; remove this line to persist.
        shutil.rmtree(self.db_path, ignore_errors=True)
        db = kuzu.Database(self.db_path)
        self._conn = kuzu.Connection(db)
        return self._conn

    def load(self, data_dir: Path = DATA_DIR) -> None:
        conn = self._connect()

        authors = json.loads((data_dir / "authors.json").read_text())
        papers = json.loads((data_dir / "papers.json").read_text())
        fields = json.loads((data_dir / "fields_of_study.json").read_text())
        methods = json.loads((data_dir / "methodologies.json").read_text())
        institutions = json.loads((data_dir / "institutions.json").read_text())

        # --- Schema (Cypher-ish DDL) ---
        conn.execute("CREATE NODE TABLE Paper(id STRING, title STRING, "
                     "abstract STRING, year INT64, field STRING, "
                     "methodology STRING, PRIMARY KEY(id))")
        conn.execute("CREATE NODE TABLE Author(id STRING, name STRING, "
                      "institution STRING, PRIMARY KEY(id))")
        conn.execute("CREATE NODE TABLE FieldOfStudy(name STRING, PRIMARY KEY(name))")
        conn.execute("CREATE NODE TABLE Methodology(name STRING, PRIMARY KEY(name))")
        conn.execute("CREATE NODE TABLE Institution(name STRING, PRIMARY KEY(name))")
        conn.execute("CREATE REL TABLE CITES(FROM Paper TO Paper)")
        conn.execute("CREATE REL TABLE AUTHORED(FROM Author TO Paper)")
        conn.execute("CREATE REL TABLE USES_METHOD(FROM Paper TO Methodology)")
        conn.execute("CREATE REL TABLE BELONGS_TO(FROM Paper TO FieldOfStudy)")

        # --- Data load ---
        for f in fields:
            conn.execute("CREATE (:FieldOfStudy {name: $n})", {"n": f})
        for m in methods:
            conn.execute("CREATE (:Methodology {name: $n})", {"n": m})
        for inst in institutions:
            conn.execute("CREATE (:Institution {name: $n})", {"n": inst})
        for a in authors:
            conn.execute(
                "CREATE (:Author {id: $id, name: $name, institution: $inst})",
                {"id": a["id"], "name": a["name"], "inst": a["institution"]},
            )
        for p in papers:
            conn.execute(
                "CREATE (:Paper {id: $id, title: $title, abstract: $abstract, "
                "year: $year, field: $field, methodology: $method})",
                {
                    "id": p["id"], "title": p["title"], "abstract": p["abstract"],
                    "year": p["year"], "field": p["field"], "method": p["methodology"],
                },
            )
        for p in papers:
            conn.execute(
                "MATCH (p:Paper {id: $pid}), (f:FieldOfStudy {name: $f}) "
                "CREATE (p)-[:BELONGS_TO]->(f)",
                {"pid": p["id"], "f": p["field"]},
            )
            conn.execute(
                "MATCH (p:Paper {id: $pid}), (m:Methodology {name: $m}) "
                "CREATE (p)-[:USES_METHOD]->(m)",
                {"pid": p["id"], "m": p["methodology"]},
            )
            for author_id in p["author_ids"]:
                conn.execute(
                    "MATCH (a:Author {id: $aid}), (p:Paper {id: $pid}) "
                    "CREATE (a)-[:AUTHORED]->(p)",
                    {"aid": author_id, "pid": p["id"]},
                )
            for cited_id in p["citation_ids"]:
                conn.execute(
                    "MATCH (p:Paper {id: $pid}), (c:Paper {id: $cid}) "
                    "CREATE (p)-[:CITES]->(c)",
                    {"pid": p["id"], "cid": cited_id},
                )
        self._loaded = True
        print(f"[graph_store:Kuzu] Loaded {len(papers)} papers, {len(authors)} authors "
              f"into on-disk DB at {self.db_path}")

    def shortest_coauthorship_path(self, author_a: str, author_b: str) -> Optional[list[str]]:
        # Kùzu supports variable-length shortest path via Cypher SHORTEST syntax.
        query = (
            "MATCH p = SHORTEST (a1:Author {id: $a})-[:AUTHORED*2..10]-(a2:Author {id: $b}) "
            "RETURN p"
        )
        result = self._conn.execute(query, {"a": author_a, "b": author_b})
        if not result.has_next():
            return None
        row = result.get_next()
        # Row parsing is driver-version dependent; callers should treat this
        # as best-effort and fall back to NetworkXGraphStore for exact parity.
        return [str(x) for x in row]

    def citation_chain(self, paper_id: str, max_depth: int = 3) -> list[list[str]]:
        query = (
            f"MATCH p = (start:Paper {{id: $pid}})-[:CITES*1..{max_depth}]->(end:Paper) "
            "RETURN [n IN nodes(p) | n.id] AS path"
        )
        result = self._conn.execute(query, {"pid": paper_id})
        paths = []
        while result.has_next():
            paths.append(result.get_next()[0])
        return paths

    def shared_methodology_papers(self, paper_id: str) -> list[str]:
        query = (
            "MATCH (p:Paper {id: $pid})-[:USES_METHOD]->(m:Methodology)"
            "<-[:USES_METHOD]-(other:Paper) WHERE other.id <> $pid "
            "RETURN DISTINCT other.id"
        )
        result = self._conn.execute(query, {"pid": paper_id})
        out = []
        while result.has_next():
            out.append(result.get_next()[0])
        return out

    def paper_neighbors_for_viz(self, paper_id: str, hops: int = 2) -> dict[str, Any]:
        raise NotImplementedError(
            "Visualization export is implemented for NetworkXGraphStore. "
            "For Kùzu, query subgraphs via Cypher and adapt to the same "
            "{'nodes': [...], 'edges': [...]} shape."
        )

    def degree_centrality(self) -> dict[str, float]:
        raise NotImplementedError(
            "Compute via Cypher aggregation (COUNT of incoming CITES edges "
            "per Paper) or export to NetworkX for exact parity with the "
            "default engine."
        )


def get_default_graph_store() -> BaseGraphStore:
    """Factory: returns the embedded NetworkX store used across the demo app."""
    store = NetworkXGraphStore()
    store.load()
    return store


if __name__ == "__main__":
    gs = get_default_graph_store()
    pid = gs.all_paper_ids()[-1]
    print(f"\nCitation chains from {pid} (depth<=3):")
    for chain in gs.citation_chain(pid, max_depth=3)[:5]:
        print("  ", " -> ".join(chain))
    print(f"\nPapers sharing methodology with {pid}: {gs.shared_methodology_papers(pid)}")
    print(f"\nTop-3 most cited papers (in-degree centrality):")
    centrality = gs.degree_centrality()
    for paper_id, score in sorted(centrality.items(), key=lambda x: -x[1])[:3]:
        print(f"  {paper_id}: {score:.3f}")
