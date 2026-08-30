# -*- coding: utf-8 -*-
"""
SSKG — the pipeline (Fig. 1 of the paper).

The stream is replayed window by window, and for every window the three phases
of the paper are executed:

* **Feeding**   — label the posts with subjects (Section 3.1), turn each post
  into a knowledge graph (Section 3.2.2), and merge it into the stream graph of
  every subject it belongs to (Section 3.2.3, Algorithms 2-3).
* **Analysis**  — score every node and edge (Section 3.3, Eq. 1-8) and forget
  the ones below ``t_d``.
* **Detection** — cluster each subject's stream graph, describe every cluster
  (Section 3.4.1) and map the subject labels onto the main stream
  (Section 3.4.2, Algorithm 4).

Everything expensive is cached: subject scores (``cache/``), knowledge graphs
(``KnowledgeGraphs/``) and phrase embeddings (``cache/``).  A second run of the
same configuration therefore skips both the classifier and the LLM entirely.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import SSKGConfig, SUBJECTS
from .data import Window, load_stream, stream_stats
from .detection import FinalEventRegistry, SubjectEvent, detect_events
from .embeddings import build_embedder
from .kg_extraction import KnowledgeGraphStore, ensure_knowledge_graphs
from .output import (save_events_report, save_post_assignments, save_topics,
                     stamped_path)
from .scoring import score_all, survivors
from .stream_graph import StreamGraph
from .subject_stream import (assign_subjects, calibrate_thresh_ps,
                             compute_subject_scores, subject_statistics)


@dataclass
class RunResult:
    topics_path: str
    assignments_path: str
    events_path: str
    registry: FinalEventRegistry
    stats: Dict[str, object]


def run(cfg: SSKGConfig) -> RunResult:
    t_start = time.time()
    log = print if cfg.verbose else (lambda *a, **k: None)

    # ---------------------------------------------------------------- data
    windows = load_stream(cfg.data_file, cfg.window_hours, cfg.max_windows)
    posts = [p for w in windows for p in w.posts]
    log(f"[data] {stream_stats(windows)}")

    # --------------------------------------------- Section 3.1: subjects
    scores = compute_subject_scores(posts, cfg)
    thresh_ps = cfg.thresh_ps
    if cfg.calibrate_thresh_ps:
        thresh_ps = calibrate_thresh_ps(scores)
        log(f"[subject] calibrated Thresh_PS = {thresh_ps:.4f}")
    assignments = assign_subjects(scores, thresh_ps)
    subject_of_post = {p.pid: subs for p, subs in zip(posts, assignments)}
    stats = subject_statistics(assignments)
    log("[subject] posts per subject: " +
        ", ".join(f"{k}={v}" for k, v in stats.items()))

    # ------------------------------------- Section 3.2.2: knowledge graphs
    kept = [p for p in posts if subject_of_post[p.pid]]
    log(f"[kg] {len(kept)} posts carry at least one subject "
        f"({len(posts) - len(kept)} dropped as noise)")
    store = ensure_knowledge_graphs(kept, cfg)

    # ------------------------------------------------------- stream graphs
    embedder = build_embedder(cfg)
    graphs: Dict[str, StreamGraph] = {
        s: StreamGraph(s, embedder, cfg.t_n, cfg.t_e) for s in SUBJECTS
    }
    registry = FinalEventRegistry(cfg.thresh_ts)

    topic_rows: List[Tuple[int, List[str]]] = []
    window_assignments: List[Tuple[int, List[Tuple[int, List[int]]]]] = []
    n_fed = n_triples = 0

    for w_i, window in enumerate(windows, start=1):
        t_win = time.time()

        # ---- feeding ----------------------------------------------------
        for post in window.posts:
            subs = subject_of_post[post.pid]
            if not subs:
                continue
            triples = store.get(post.pid)
            if not triples:
                continue
            for s in subs:
                n_triples += graphs[SUBJECTS[s]].feed(
                    triples, post.timestamp, post.pid)
            n_fed += 1

        now = max(p.timestamp for p in window.posts)

        # ---- analysis, then detection (Fig. 7) ---------------------------
        # The paper's order is: score the stream graph, forget the nodes below
        # t_d, and only then cluster what is left — so an "event" is always a
        # cluster of *bursty* entities.
        do_analysis = (w_i % max(1, cfg.forget_every) == 0)
        window_events: List[Tuple[SubjectEvent, int]] = []
        for subject, graph in graphs.items():
            if len(graph.nodes) == 0:
                continue
            node_scores = score_all([graph.nodes.times[i] for i in range(len(graph.nodes))],
                                    now, cfg.ts_seconds, cfg.weight_min, cfg.weight_max)
            edge_scores = score_all([graph.edges.times[i] for i in range(len(graph.edges))],
                                    now, cfg.ts_seconds, cfg.weight_min, cfg.weight_max)

            if do_analysis:                       # forgetting mechanism
                keep_n = survivors(node_scores, cfg.t_d)
                keep_e = survivors(edge_scores, cfg.t_d)
                graph.prune(keep_n, keep_e)
                # the survivors keep their occurrence times, so their scores
                # are unchanged — just re-index them
                node_scores = node_scores[keep_n] if len(keep_n) else node_scores[:0]
                edge_scores = edge_scores[keep_e] if len(keep_e) else edge_scores[:0]
                if len(graph.nodes) == 0:
                    continue

            events = detect_events(
                graph, node_scores, edge_scores, window.number,
                method=cfg.clustering, seed=cfg.seed,
                min_event_size=cfg.min_event_size,
                max_titles=cfg.titles_per_event,
                spectral_clusters=cfg.spectral_clusters,
                hcluster_distance=cfg.hcluster_distance,
            )
            window_events.extend(registry.add_all(events))

        # ---- record ------------------------------------------------------
        post_events: Dict[int, List[int]] = defaultdict(list)
        for ev, number in window_events:
            topic_rows.append((window.number, ev.all_strings))
            for pid in ev.post_ids:
                if number not in post_events[pid]:
                    post_events[pid].append(number)

        window_assignments.append(
            (window.number,
             [(p.pid, post_events.get(p.pid, [])) for p in window.posts])
        )

        if cfg.verbose:
            sizes = ", ".join(f"{s}:{len(g.nodes)}" for s, g in graphs.items()
                              if len(g.nodes))
            print(f"[win {window.number:>3}] {len(window.posts):>4} posts | "
                  f"{len(window_events):>3} events | graph nodes {{{sizes}}} | "
                  f"{time.time() - t_win:.1f}s")

    # ------------------------------------------------------------- output
    embedder.save_cache()
    stamp = cfg.stamp()
    topics_path = save_topics(stamped_path(cfg.out_dir, "Topic_Systemresult", stamp),
                              topic_rows)
    assignments_path = save_post_assignments(
        stamped_path(cfg.out_dir, "ResultsToCompaire", stamp), window_assignments)
    events_path = save_events_report(
        stamped_path(cfg.out_dir, "FinalEvents", stamp, ".tsv"), registry.events)

    elapsed = time.time() - t_start
    run_stats = {
        "posts": len(posts),
        "posts_fed": n_fed,
        "triples_fed": n_triples,
        "windows": len(windows),
        "final_events": len(registry.events),
        "subject_counts": stats,
        "thresh_ps": thresh_ps,
        "elapsed_seconds": round(elapsed, 1),
    }
    log(f"[done] {len(registry.events)} final events, "
        f"{n_triples} triples fed, {elapsed:.1f}s")
    log(f"[done] {topics_path}")
    log(f"[done] {assignments_path}")
    return RunResult(topics_path, assignments_path, events_path, registry, run_stats)
