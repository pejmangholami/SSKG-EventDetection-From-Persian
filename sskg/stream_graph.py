# -*- coding: utf-8 -*-
"""
SSKG — Section 3.2.3: the *stream graph* (Algorithms 2 and 3).

The stream graph is a knowledge graph enriched with the temporal structure of
the stream.  For every subject stream one stream graph is maintained; feeding a
post means merging that post's knowledge graph into it (Algorithm 2), where
each node and edge phrase is embedded and either merged into the nearest
existing entity or inserted as a new one (Algorithm 3, ``UpSert``).

Every entity (node **and** edge — Fig. 3) stores:

* every surface form it has absorbed, with counts (``titles``),
* the **timestamp of each occurrence** (this is what makes the graph a
  *stream* graph and what Section 3.3 scores),
* the ids of the posts it came from (used in Section 3.4.1 to map an event
  back to its posts).

Complexity note
---------------
Algorithm 3 as printed re-embeds the whole entity list on every UpSert, which
is O(N²) embeddings.  This implementation is mathematically identical but
practical:

* every phrase is embedded **once** and cached (see :mod:`sskg.embeddings`);
* an exact-string fast path skips the search for phrases already seen;
* the nearest neighbour is found with a single BLAS matrix-vector product over
  a pre-allocated, L2-normalised matrix of entity vectors, so cosine distance
  is ``1 - V @ v``;
* the representative vector of a merged entity is the running mean of its
  members (re-normalised), so merging never needs the members again.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .embeddings import BaseEmbedder, normalise_phrase


class EntityIndex:
    """A growing set of merged phrases with occurrence times (nodes or edges)."""

    def __init__(self, dim: int, merge_threshold: float, capacity: int = 1024):
        self.dim = dim
        self.merge_threshold = float(merge_threshold)
        self._vectors = np.zeros((capacity, dim), dtype=np.float32)
        self._sums = np.zeros((capacity, dim), dtype=np.float32)
        self.size = 0
        self.titles: List[Counter] = []
        self.times: List[List[float]] = []
        self.posts: List[set] = []
        self.count: List[int] = []
        self._exact: Dict[str, int] = {}     # surface form -> entity index

    # -- internals ----------------------------------------------------------
    def _grow(self) -> None:
        cap = self._vectors.shape[0]
        if self.size < cap:
            return
        new = max(cap * 2, 1024)
        v = np.zeros((new, self.dim), dtype=np.float32)
        s = np.zeros((new, self.dim), dtype=np.float32)
        v[:cap] = self._vectors
        s[:cap] = self._sums
        self._vectors, self._sums = v, s

    @property
    def vectors(self) -> np.ndarray:
        """(size, dim) L2-normalised representative vectors."""
        return self._vectors[:self.size]

    # -- Algorithm 3 --------------------------------------------------------
    def upsert(self, phrase: str, vec: np.ndarray, when: float,
               post_id: Optional[int] = None) -> int:
        """Insert `phrase` or merge it into the nearest entity; return its index.

        `vec` must be L2 normalised.  Merging happens when the cosine distance
        to the nearest entity is below ``merge_threshold`` (``t_n``/``t_e``).
        """
        phrase = normalise_phrase(phrase)
        if not phrase:
            return -1

        idx = self._exact.get(phrase)
        if idx is None:
            if self.size:
                sims = self.vectors @ vec                # cosine similarity
                best = int(np.argmax(sims))
                if 1.0 - float(sims[best]) < self.merge_threshold:
                    idx = best
            if idx is None:                              # brand-new entity
                self._grow()
                idx = self.size
                self._vectors[idx] = vec
                self._sums[idx] = vec
                self.size += 1
                self.titles.append(Counter())
                self.times.append([])
                self.posts.append(set())
                self.count.append(0)
            self._exact[phrase] = idx

        if phrase not in self.titles[idx]:               # first time here
            self._sums[idx] += vec
            n = float(np.linalg.norm(self._sums[idx]))
            if n > 0:
                self._vectors[idx] = self._sums[idx] / n
        self.titles[idx][phrase] += 1
        self.times[idx].append(float(when))
        self.count[idx] += 1
        if post_id is not None:
            self.posts[idx].add(int(post_id))
        return idx

    # -- queries ------------------------------------------------------------
    def title(self, idx: int) -> str:
        """The most frequent surface form of an entity."""
        if 0 <= idx < self.size and self.titles[idx]:
            return self.titles[idx].most_common(1)[0][0]
        return ""

    def all_titles(self, idx: int) -> List[str]:
        return [t for t, _ in self.titles[idx].most_common()]

    def keep(self, keep_idx: Sequence[int]) -> Dict[int, int]:
        """Drop every entity not in `keep_idx` (the forgetting mechanism).

        Returns the mapping old index -> new index for the survivors.
        """
        keep_idx = sorted(set(int(i) for i in keep_idx if 0 <= i < self.size))
        remap = {old: new for new, old in enumerate(keep_idx)}
        n = len(keep_idx)
        vectors = self._vectors[keep_idx].copy() if n else np.zeros((0, self.dim), np.float32)
        sums = self._sums[keep_idx].copy() if n else np.zeros((0, self.dim), np.float32)
        cap = max(n, 1024)
        self._vectors = np.zeros((cap, self.dim), dtype=np.float32)
        self._sums = np.zeros((cap, self.dim), dtype=np.float32)
        if n:
            self._vectors[:n] = vectors
            self._sums[:n] = sums
        self.titles = [self.titles[i] for i in keep_idx]
        self.times = [self.times[i] for i in keep_idx]
        self.posts = [self.posts[i] for i in keep_idx]
        self.count = [self.count[i] for i in keep_idx]
        self.size = n
        self._exact = {p: remap[i] for p, i in self._exact.items() if i in remap}
        return remap

    def __len__(self) -> int:
        return self.size


class StreamGraph:
    """The stream graph of ONE subject stream (Fig. 3, Algorithms 2-3)."""

    def __init__(self, subject: str, embedder: BaseEmbedder,
                 t_n: float = 0.10, t_e: float = 0.10):
        self.subject = subject
        self.embedder = embedder
        dim = embedder.dim
        self.nodes = EntityIndex(dim, t_n)
        self.edges = EntityIndex(dim, t_e)
        #: relations as (node1, edge, node2, timestamp, post_id)
        self.relations: List[Tuple[int, int, int, float, int]] = []

    # -- Algorithm 2 --------------------------------------------------------
    def feed(self, triples: Sequence[Tuple[str, str, str]],
             when: float, post_id: int) -> int:
        """Merge one post's knowledge graph into the stream graph."""
        triples = [t for t in triples if len(t) == 3]
        if not triples:
            return 0
        phrases: List[str] = []
        for h, r, o in triples:
            phrases.extend((h, r, o))
        vectors = self.embedder.encode(phrases)          # one batched call
        added = 0
        for k, (h, r, o) in enumerate(triples):
            v_h, v_r, v_o = vectors[3 * k], vectors[3 * k + 1], vectors[3 * k + 2]
            n1 = self.nodes.upsert(h, v_h, when, post_id)
            n2 = self.nodes.upsert(o, v_o, when, post_id)
            e = self.edges.upsert(r, v_r, when, post_id)
            if n1 < 0 or n2 < 0 or e < 0 or n1 == n2:
                continue
            self.relations.append((n1, e, n2, float(when), int(post_id)))
            added += 1
        return added

    # -- forgetting ---------------------------------------------------------
    def prune(self, keep_nodes: Sequence[int], keep_edges: Sequence[int]) -> None:
        """Apply the forgetting mechanism (Section 3.3) to nodes and edges."""
        node_map = self.nodes.keep(keep_nodes)
        edge_map = self.edges.keep(keep_edges)
        self.relations = [
            (node_map[n1], edge_map[e], node_map[n2], t, p)
            for (n1, e, n2, t, p) in self.relations
            if n1 in node_map and n2 in node_map and e in edge_map
        ]

    # -- views --------------------------------------------------------------
    def node_posts(self, idx: int) -> set:
        return self.nodes.posts[idx]

    def triples_of(self, node_idx: Iterable[int]) -> List[Tuple[int, int, int]]:
        """Relations whose two endpoints are both inside `node_idx`."""
        wanted = set(int(i) for i in node_idx)
        return [(n1, e, n2) for (n1, e, n2, _, _) in self.relations
                if n1 in wanted and n2 in wanted]

    def summary(self) -> str:
        return (f"[{self.subject}] {len(self.nodes)} nodes, "
                f"{len(self.edges)} edges, {len(self.relations)} relations")


def pairwise_edge_weights(graph: StreamGraph,
                          edge_scores: np.ndarray) -> Dict[Tuple[int, int], float]:
    """Weight of every node pair = max score of the relations that connect them.

    Section 4.3 of the paper clusters "weighted stream graphs (where the weight
    of each edge is calculated according to Equation (8))", i.e. the burstiness
    score of the *relation* phrase that links the two nodes.
    """
    weights: Dict[Tuple[int, int], float] = defaultdict(float)
    for (n1, e, n2, _, _) in graph.relations:
        if n1 == n2:
            continue
        key = (n1, n2) if n1 < n2 else (n2, n1)
        w = float(edge_scores[e]) if e < len(edge_scores) else 0.0
        if w > weights[key]:
            weights[key] = w
    return dict(weights)
