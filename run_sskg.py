#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SSKG — run the full event-detection pipeline.

    python run_sskg.py                       # paper configuration
    python run_sskg.py --offline             # no downloads, no API key
    python run_sskg.py --max-windows 4       # quick smoke run

If the knowledge graphs of the posts are already in ``KnowledgeGraphs/`` they
are loaded; otherwise they are built first and stored there, and only then does
the rest of the pipeline start.  Run ``build_knowledge_graphs.py`` if you want
that step on its own.
"""

from __future__ import annotations

import argparse
import json
import sys

from sskg.config import SSKGConfig
from sskg.pipeline import run


def build_parser() -> argparse.ArgumentParser:
    d = SSKGConfig()
    p = argparse.ArgumentParser(
        description="SSKG — subject stream knowledge graph event detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    g = p.add_argument_group("data")
    g.add_argument("--data-file", default=d.data_file)
    g.add_argument("--window-hours", type=int, default=d.window_hours,
                   help="detection/window granularity (12 = gold standard)")
    g.add_argument("--max-windows", type=int, default=None,
                   help="process only the first N windows (smoke test)")

    g = p.add_argument_group("subject stream (Section 3.1)")
    g.add_argument("--subject-backend", choices=["parsbert", "keyword"],
                   default=d.subject_backend)
    g.add_argument("--subject-model", default=d.subject_model)
    g.add_argument("--thresh-ps", type=float, default=d.thresh_ps,
                   help="Thresh_PS of Algorithm 1")
    g.add_argument("--subject-score-mode", choices=["softmax", "sigmoid"],
                   default=d.subject_score_mode,
                   help="how ParsBERT logits become L_score; 'sigmoid' allows "
                        "one post to reach several subjects at Thresh_PS >= 0.5")
    g.add_argument("--calibrate-thresh-ps", action="store_true",
                   help="fit Thresh_PS so that 795 posts stay unlabelled (Fig. 8)")

    g = p.add_argument_group("knowledge graphs (Section 3.2.2)")
    g.add_argument("--kg-backend", choices=["openai", "anthropic", "heuristic"],
                   default=d.kg_backend)
    g.add_argument("--kg-model", default=d.kg_model)
    g.add_argument("--kg-dir", default=d.kg_dir)
    g.add_argument("--kg-workers", type=int, default=d.kg_workers)
    g.add_argument("--kg-temperature", type=float, default=d.kg_temperature,
                   help="omitted by default, so the API uses its own default")

    g = p.add_argument_group("embeddings (Section 3.2.3)")
    g.add_argument("--embed-backend", choices=["parsbert", "hashing"],
                   default=d.embed_backend)
    g.add_argument("--embed-model", default=d.embed_model)
    g.add_argument("--embed-batch-size", type=int, default=d.embed_batch_size)

    g = p.add_argument_group("stream graph / forgetting (Sections 3.2.3, 3.3)")
    g.add_argument("--t-n", type=float, default=d.t_n,
                   help="node merge threshold (cosine distance)")
    g.add_argument("--t-e", type=float, default=d.t_e,
                   help="edge merge threshold (cosine distance)")
    g.add_argument("--t-d", type=float, default=d.t_d,
                   help="forgetting threshold on the node score")
    g.add_argument("--ts-seconds", type=int, default=d.ts_seconds,
                   help="TS: length of the NAP/NAF time slot, in seconds")
    g.add_argument("--forget-every", type=int, default=d.forget_every)

    g = p.add_argument_group("detection (Section 3.4)")
    g.add_argument("--clustering", choices=["louvain", "spectral", "hcluster"],
                   default=d.clustering)
    g.add_argument("--thresh-ts", type=float, default=d.thresh_ts,
                   help="Thresh_ts of Algorithm 4 (label merging): Euclidean "
                        "distance between un-normalised title embeddings")
    g.add_argument("--titles-per-event", type=int, default=d.titles_per_event)
    g.add_argument("--min-event-size", type=int, default=d.min_event_size)

    g = p.add_argument_group("runtime")
    g.add_argument("--out-dir", default=d.out_dir)
    g.add_argument("--seed", type=int, default=d.seed)
    g.add_argument("--quiet", action="store_true")
    g.add_argument("--offline", action="store_true",
                   help="shortcut for --subject-backend keyword "
                        "--kg-backend heuristic --embed-backend hashing")
    return p


def config_from_args(args) -> SSKGConfig:
    cfg = SSKGConfig(
        data_file=args.data_file,
        window_hours=args.window_hours,
        max_windows=args.max_windows,
        subject_backend=args.subject_backend,
        subject_model=args.subject_model,
        thresh_ps=args.thresh_ps,
        subject_score_mode=args.subject_score_mode,
        calibrate_thresh_ps=args.calibrate_thresh_ps,
        kg_backend=args.kg_backend,
        kg_model=args.kg_model,
        kg_dir=args.kg_dir,
        kg_workers=args.kg_workers,
        kg_temperature=args.kg_temperature,
        embed_backend=args.embed_backend,
        embed_model=args.embed_model,
        embed_batch_size=args.embed_batch_size,
        t_n=args.t_n,
        t_e=args.t_e,
        t_d=args.t_d,
        ts_seconds=args.ts_seconds,
        forget_every=args.forget_every,
        clustering=args.clustering,
        thresh_ts=args.thresh_ts,
        titles_per_event=args.titles_per_event,
        min_event_size=args.min_event_size,
        out_dir=args.out_dir,
        seed=args.seed,
        verbose=not args.quiet,
    )
    if args.offline:
        cfg.subject_backend = "keyword"
        cfg.kg_backend = "heuristic"
        cfg.embed_backend = "hashing"
    return cfg


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)
    if cfg.verbose:
        print("[config] " + json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2))
    result = run(cfg)
    print(json.dumps(result.stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
