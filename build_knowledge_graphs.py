#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SSKG — build (or complete) the per-post knowledge graphs, and nothing else.

    export OPENAI_API_KEY=...
    python build_knowledge_graphs.py                    # all posts
    python build_knowledge_graphs.py --limit 200        # try it on 200 posts
    python build_knowledge_graphs.py --backend heuristic --offline-subjects
    python build_knowledge_graphs.py --show 5           # print a few graphs

The store (``KnowledgeGraphs/knowledge_graphs.jsonl``) is append-only and
resumable: interrupt it whenever you like and run it again — only the posts
that are still missing are sent to the model.  ``run_sskg.py`` calls exactly
the same code, so building here first is an optimisation, not a requirement.
"""

from __future__ import annotations

import argparse
import sys

from sskg.config import SSKGConfig
from sskg.data import load_stream
from sskg.kg_extraction import KnowledgeGraphStore, ensure_knowledge_graphs
from sskg.subject_stream import (assign_subjects, calibrate_thresh_ps,
                                 compute_subject_scores)


def main(argv=None) -> int:
    d = SSKGConfig()
    p = argparse.ArgumentParser(
        description="Build the per-post knowledge graphs (Section 3.2.2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-file", default=d.data_file)
    p.add_argument("--kg-dir", default=d.kg_dir)
    p.add_argument("--backend", choices=["openai", "anthropic", "heuristic"],
                   default=d.kg_backend)
    p.add_argument("--model", default=d.kg_model)
    p.add_argument("--workers", type=int, default=d.kg_workers)
    p.add_argument("--limit", type=int, default=None,
                   help="only the first N posts (chronologically)")
    p.add_argument("--subject-filter", action="store_true",
                   help="only extract for posts that reach a subject "
                        "(what the pipeline actually needs)")
    p.add_argument("--subject-backend", choices=["parsbert", "keyword"],
                   default=d.subject_backend)
    p.add_argument("--thresh-ps", type=float, default=d.thresh_ps)
    p.add_argument("--show", type=int, default=0,
                   help="print N stored graphs and exit")
    args = p.parse_args(argv)

    cfg = SSKGConfig(data_file=args.data_file, kg_dir=args.kg_dir,
                     kg_backend=args.backend, kg_model=args.model,
                     kg_workers=args.workers,
                     subject_backend=args.subject_backend,
                     thresh_ps=args.thresh_ps)

    windows = load_stream(cfg.data_file, cfg.window_hours)
    posts = [p_ for w in windows for p_ in w.posts]
    if args.limit:
        posts = posts[: args.limit]

    if args.show:
        store = KnowledgeGraphStore(cfg.kg_dir)
        print(store.stats())
        shown = 0
        for post in posts:
            triples = store.get(post.pid)
            if not triples:
                continue
            print("=" * 78)
            print(f"post {post.pid} @ {post.dt}")
            print(post.text[:400])
            for t in triples:
                print("   ", t)
            shown += 1
            if shown >= args.show:
                break
        return 0

    if args.subject_filter:
        scores = compute_subject_scores(posts, cfg)
        assign = assign_subjects(scores, cfg.thresh_ps)
        posts = [p_ for p_, s in zip(posts, assign) if s]
        print(f"[kg] {len(posts)} posts left after the subject filter")

    ensure_knowledge_graphs(posts, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
