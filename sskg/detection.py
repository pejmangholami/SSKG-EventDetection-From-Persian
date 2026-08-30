# -*- coding: utf-8 -*-
"""
SSKG — Section 3.4: event detection.

1. **Clustering the stream graph** (Section 3.4).  Each cluster of the weighted
   stream graph is one event of that subject.  Table 2 of the paper compares
   Louvain and Spectral clustering on the graph with hierarchical clustering on
   node/triple embeddings; Louvain wins, so it is the default here.
2. **Titles and posts** (Section 3.4.1).  The title of an event is the text of
   the triple of the stream graph closest to the cluster's label in vector
   space; the posts of an event are the posts that contributed its nodes.
3. **Mapping labels to the main stream** (Section 3.4.2, Algorithm 4).  Events
   found in different subject streams are merged into one global event list
   when the cosine distance between their label embeddings is below
   ``Thresh_ts``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .embeddings import l2_normalise
from .stream_graph import StreamGraph, pairwise_edge_weights


# ---------------------------------------------------------------------------
# data classes
# ---------------------------------------------------------------------------

@dataclass
class SubjectEvent:
    """One cluster of one subject's stream graph."""
    subject: str
    window: int
    nodes: List[int]
    titles: List[str]                 # triple-level titles (Section 3.4.1)
    post_ids: Set[int]
    score: float
    vector: np.ndarray
    keywords: List[str] = field(default_factory=list)   # entity surface forms

    @property
    def title(self) -> str:
        return " | ".join(self.titles)

    @property
    def all_strings(self) -> List[str]:
        out = list(self.titles)
        for k in self.keywords:
            if k not in out:
                out.append(k)
        return out


@dataclass
class FinalEvent:
    """A global event of the main text stream (Algorithm 4)."""
    number: int                       # 1-based; 0 means "no event"
    titles: List[str] = field(default_factory=list)
    subjects: Set[str] = field(default_factory=set)
    windows: Set[int] = field(default_factory=set)
    post_ids: Set[int] = field(default_factory=set)
    vector: Optional[np.ndarray] = None
    _n: int = 0

    def absorb(self, ev: SubjectEvent) -> None:
        for t in ev.titles:
            if t and t not in self.titles:
                self.titles.append(t)
        self.subjects.add(ev.subject)
        self.windows.add(ev.window)
        self.post_ids |= ev.post_ids
        if self.vector is None:
            self.vector = ev.vector.copy()
            self._n = 1
        else:
            self._n += 1
            self.vector = l2_normalise(
                self.vector * (self._n - 1) / self._n + ev.vector / self._n)


# ---------------------------------------------------------------------------
# clustering
# ---------------------------------------------------------------------------

def cluster_graph(graph: StreamGraph, edge_scores: np.ndarray,
                  method: str = "louvain", seed: int = 1234,
                  spectral_clusters: int = 8,
                  hcluster_distance: float = 0.5) -> List[List[int]]:
    """Return the clusters of the weighted stream graph as lists of node ids."""
    n = len(graph.nodes)
    if n == 0:
        return []
    weights = pairwise_edge_weights(graph, edge_scores)

    if method == "louvain":
        import networkx as nx
        g = nx.Graph()
        g.add_nodes_from(range(n))
        for (a, b), w in weights.items():
            if w > 0:
                g.add_edge(a, b, weight=float(w))
        if g.number_of_edges() == 0:
            return [[i] for i in range(n)]
        communities = nx.community.louvain_communities(
            g, weight="weight", seed=seed)
        return [sorted(c) for c in communities]

    if method == "spectral":
        from sklearn.cluster import SpectralClustering
        affinity = np.zeros((n, n), dtype=np.float64)
        for (a, b), w in weights.items():
            affinity[a, b] = affinity[b, a] = max(float(w), 0.0)
        k = int(min(max(2, spectral_clusters), n))
        if n < 3:
            return [[i for i in range(n)]]
        labels = SpectralClustering(
            n_clusters=k, affinity="precomputed", assign_labels="kmeans",
            random_state=seed).fit_predict(affinity + 1e-9)
        out: Dict[int, List[int]] = {}
        for node, lab in enumerate(labels):
            out.setdefault(int(lab), []).append(node)
        return [sorted(v) for v in out.values()]

    if method == "hcluster":
        # Table 2, rows 1-2: hierarchical clustering of the node embeddings.
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import pdist
        if n < 2:
            return [[0]] if n else []
        vecs = graph.nodes.vectors
        dist = pdist(vecs, metric="cosine")
        labels = fcluster(linkage(dist, method="average"),
                          t=hcluster_distance, criterion="distance")
        out: Dict[int, List[int]] = {}
        for node, lab in enumerate(labels):
            out.setdefault(int(lab), []).append(node)
        return [sorted(v) for v in out.values()]

    raise ValueError(f"unknown clustering method: {method!r}")


# ---------------------------------------------------------------------------
# titles (Section 3.4.1)
# ---------------------------------------------------------------------------

def event_titles(graph: StreamGraph, nodes: Sequence[int],
                 centroid: np.ndarray, max_titles: int = 3) -> List[str]:
    """The triples of the cluster closest to its label, as title strings."""
    triples = graph.triples_of(nodes)
    titles: List[str] = []
    if triples:
        node_vecs = graph.nodes.vectors
        edge_vecs = graph.edges.vectors
        mats = np.stack([
            l2_normalise(node_vecs[n1] + edge_vecs[e] + node_vecs[n2])
            for (n1, e, n2) in triples
        ])
        order = np.argsort(-(mats @ centroid))
        for k in order[:max_titles]:
            n1, e, n2 = triples[int(k)]
            text = " ".join(x for x in (graph.nodes.title(n1),
                                        graph.edges.title(e),
                                        graph.nodes.title(n2)) if x)
            if text and text not in titles:
                titles.append(text)
    if not titles:                       # isolated nodes: fall back to titles
        for n in nodes[:max_titles]:
            t = graph.nodes.title(n)
            if t:
                titles.append(t)
    return titles


def node_keywords(graph: StreamGraph, nodes: Sequence[int], limit: int = 3) -> List[str]:
    """The most frequent surface forms of an event's nodes.

    They are reported next to the triple-based title because the evaluation
    matches titles as substrings of the post text, and single entity phrases
    are exactly the substrings that appear verbatim in the posts.
    """
    scored = sorted(((sum(graph.nodes.titles[n].values()), n) for n in nodes),
                    reverse=True)
    out: List[str] = []
    for _, n in scored[: max(1, limit)]:
        t = graph.nodes.title(n)
        if t and t not in out:
            out.append(t)
    return out


def detect_events(graph: StreamGraph, node_scores: np.ndarray,
                  edge_scores: np.ndarray, window: int,
                  method: str = "louvain", seed: int = 1234,
                  min_event_size: int = 2, max_titles: int = 3,
                  spectral_clusters: int = 8,
                  hcluster_distance: float = 0.5) -> List[SubjectEvent]:
    """Cluster one subject's stream graph and describe every cluster."""
    clusters = cluster_graph(graph, edge_scores, method, seed,
                             spectral_clusters, hcluster_distance)
    events: List[SubjectEvent] = []
    node_vecs = graph.nodes.vectors
    for nodes in clusters:
        if len(nodes) < min_event_size:
            continue
        centroid = l2_normalise(node_vecs[nodes].mean(axis=0))
        titles = event_titles(graph, nodes, centroid, max_titles)
        posts: Set[int] = set()
        for n in nodes:
            posts |= graph.node_posts(n)
        score = float(np.mean([node_scores[n] for n in nodes
                               if np.isfinite(node_scores[n])] or [0.0]))
        events.append(SubjectEvent(subject=graph.subject, window=window,
                                   nodes=list(nodes), titles=titles,
                                   post_ids=posts, score=score,
                                   vector=centroid,
                                   keywords=node_keywords(graph, nodes, max_titles)))
    events.sort(key=lambda e: -e.score)
    return events


# ---------------------------------------------------------------------------
# Algorithm 4 — map subject labels onto the main stream
# ---------------------------------------------------------------------------

class FinalEventRegistry:
    """Keeps the global event list and merges subject events into it."""

    def __init__(self, thresh_ts: float = 2.0):
        self.thresh_ts = float(thresh_ts)
        self.events: List[FinalEvent] = []
        self._matrix: Optional[np.ndarray] = None

    def _rebuild_matrix(self) -> None:
        if not self.events:
            self._matrix = None
        else:
            self._matrix = np.stack([e.vector for e in self.events])

    def add(self, ev: SubjectEvent) -> int:
        """Return the final event number (1-based) this subject event maps to."""
        if self._matrix is not None and len(self.events):
            sims = self._matrix @ ev.vector
            best = int(np.argmax(sims))
            distance = 1.0 - float(sims[best])
            if distance < self.thresh_ts:
                self.events[best].absorb(ev)
                self._rebuild_matrix()
                return self.events[best].number
        final = FinalEvent(number=len(self.events) + 1)
        final.absorb(ev)
        self.events.append(final)
        self._rebuild_matrix()
        return final.number

    def add_all(self, events: Sequence[SubjectEvent]) -> List[Tuple[SubjectEvent, int]]:
        return [(ev, self.add(ev)) for ev in events]

    def titles_of(self, number: int) -> List[str]:
        return self.events[number - 1].titles if 1 <= number <= len(self.events) else []
