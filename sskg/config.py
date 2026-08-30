# -*- coding: utf-8 -*-
"""
SSKG — central configuration.

Every tunable of the paper lives here, with the value reported in the paper as
the default.  Values that the paper does NOT report are marked with
`# NOT REPORTED` and carry a documented, reasonable default; they are the first
knobs to tune when reproducing the published numbers.

Paper: P. Gholami-Dastgerdi, M.-R. Feizi-Derakhshi, P. Salehpour,
"SSKG: Subject stream knowledge graph, a new approach for event detection from
text", Ain Shams Engineering Journal, 2024. doi:10.1016/j.asej.2024.103040
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional


# ---------------------------------------------------------------------------
# The eight subject groups of Section 3.1.2.
# ---------------------------------------------------------------------------
SUBJECTS: List[str] = [
    "social",            # اجتماعی
    "economic",          # اقتصادی
    "international",     # بین‌الملل
    "political",         # سیاسی
    "science_tech",      # علمی و فناوری
    "cultural_art",      # فرهنگی و هنری
    "sport",             # ورزشی
    "medical",           # پزشکی
]

SUBJECTS_FA = {
    "social": "اجتماعی",
    "economic": "اقتصادی",
    "international": "بین‌الملل",
    "political": "سیاسی",
    "science_tech": "علمی و فناوری",
    "cultural_art": "فرهنگی و هنری",
    "sport": "ورزشی",
    "medical": "پزشکی",
}

# Post counts per subject reported in Fig. 8 of the paper.  Used only by the
# optional threshold calibration (`--calibrate-subject-threshold`).
PAPER_SUBJECT_COUNTS = {
    "social": 3562,
    "economic": 690,
    "international": 610,
    "political": 3209,
    "science_tech": 744,
    "cultural_art": 767,
    "sport": 440,
    "medical": 385,
}
PAPER_UNLABELLED_POSTS = 795     # Fig. 8, row 9 ("Deleted Set")
PAPER_TOTAL_POSTS = 10284        # Fig. 8, row 10


@dataclass
class SSKGConfig:
    """All parameters of the SSKG pipeline."""

    # -- data ---------------------------------------------------------------
    data_file: str = "AllData.npy"
    #: The shipped dataset is binned into 1-hour windows; the gold standard
    #: annotates 12-hour windows.  Re-binning makes the pipeline's window
    #: numbers identical to the gold standard's (62 windows for January 2017).
    window_hours: int = 12
    #: Process only the first N windows (smoke tests). None = all.
    max_windows: Optional[int] = None

    # -- Section 3.1: subject stream ---------------------------------------
    #: Backend used to label each post with 0..8 subjects.
    #:   'parsbert'  -> HooshvareLab/bert-fa-base-uncased-clf-persiannews
    #:   'keyword'   -> offline lexicon classifier (no downloads, for tests)
    subject_backend: str = "parsbert"
    subject_model: str = "HooshvareLab/bert-fa-base-uncased-clf-persiannews"
    #: Thresh_PS of Algorithm 1.  A post joins every subject whose score is
    #: >= this value; a post that reaches no subject is dropped from all
    #: streams ("Deleted Set" of Fig. 8).
    thresh_ps: float = 0.5
    #: How ParsBERT_S turns the classifier logits into the L_score values of
    #: Algorithm 1:
    #:   'softmax' - one distribution over the eight groups (the head was
    #:               trained this way).  With Thresh_PS = 0.5 at most ONE group
    #:               can pass, so posts are effectively single-label.
    #:   'sigmoid' - one independent probability per group, i.e. a multi-label
    #:               reading; this is the setting that can reproduce the
    #:               overlapping subject sets of Fig. 8.
    subject_score_mode: str = "softmax"
    #: If True, thresh_ps is re-fitted so that the number of unlabelled posts
    #: matches PAPER_UNLABELLED_POSTS (795) on this dataset.
    calibrate_thresh_ps: bool = False

    # -- Section 3.2.2: per-post knowledge graph ---------------------------
    #: 'openai' / 'anthropic' reproduce the paper (an LLM + the Few-Shot prompt);
    #: 'heuristic' is a dependency-free offline extractor used for smoke tests.
    kg_backend: str = "openai"
    kg_model: str = "gpt-4o-mini"
    kg_dir: str = "KnowledgeGraphs"
    kg_workers: int = 8           # parallel LLM requests
    kg_max_retries: int = 5
    #: None -> do not send a temperature and let the API use its own default.
    kg_temperature: Optional[float] = None
    kg_max_triples_per_post: int = 32

    # -- embeddings (used by Alg. 3 and by Sections 3.4.1/3.4.2) -----------
    #: 'parsbert' reproduces the paper; 'hashing' is an offline character
    #: n-gram embedder that makes the whole pipeline runnable without downloads.
    embed_backend: str = "parsbert"
    embed_model: str = "HooshvareLab/bert-fa-base-uncased"
    embed_batch_size: int = 64
    embed_max_length: int = 32
    embed_dim_hashing: int = 512   # only for the 'hashing' backend
    embed_cache: str = "cache/embeddings"

    # -- Section 3.2.3 / Algorithms 2-3: stream-graph merging --------------
    #: t_n / t_e — cosine-DISTANCE thresholds below which two node/edge phrases
    #: are merged into one entity.  Edge phrases (relations) are merged more
    #: aggressively than node phrases (entities).
    t_n: float = 0.15
    t_e: float = 0.20

    # -- Section 3.3: node scoring and forgetting --------------------------
    #: TS of Eq. (2)-(4): the length of the time slot used to build the NAP and
    #: NAF vectors.  The paper's default is 60 seconds.
    ts_seconds: int = 60
    weight_min: float = 0.1       # Fig. 6: oldest interval
    weight_max: float = 1.0       # Fig. 6: newest interval
    #: t_d — nodes/edges with Score < t_d are forgotten (Table 1, Table 3).
    t_d: float = 1.2
    #: Run analysis + forgetting every N detection steps (Fig. 1 "TimeStep").
    forget_every: int = 1

    # -- Section 3.4: detection --------------------------------------------
    #: 'louvain' (best in Table 2), 'spectral' or 'hcluster'.
    clustering: str = "louvain"
    spectral_clusters: int = 8     # only for 'spectral'
    hcluster_distance: float = 0.5  # only for 'hcluster'
    #: Thresh_ts of Algorithm 4 — cosine DISTANCE below which two subject-level
    #: labels are merged into the same final event.  The paper's best value is
    #: 2 (Table 3 + Section 4.3); note that cosine distance <= 2 always, so 2
    #: merges every label into a single final event.  This is consistent with
    #: the paper's minimal class entropy (0.3544) and maximal cluster entropy.
    thresh_ts: float = 2.0
    #: Number of representative titles kept per event.
    titles_per_event: int = 3
    #: Minimum number of nodes for a cluster to be reported as an event.
    min_event_size: int = 2

    # -- runtime ------------------------------------------------------------
    out_dir: str = "SystemResults"
    seed: int = 1234
    verbose: bool = True

    # -- derived ------------------------------------------------------------
    def stamp(self) -> str:
        """Parameter stamp used in output file names (parsed by evaluate.py)."""
        return (
            f"win-{self.window_hours}"
            f"_tn-{self.t_n}"
            f"_te-{self.t_e}"
            f"_td-{self.t_d}"
            f"_ts-{self.thresh_ts}"
            f"_ps-{self.thresh_ps}"
            f"_clu-{self.clustering}"
        )

    def to_dict(self):
        return asdict(self)
