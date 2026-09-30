# -*- coding: utf-8 -*-
"""
End-to-end tests: a synthetic stream is written as an `AllData.npy`, run
through the whole pipeline, and the produced result files are read back.

They cover what the unit tests cannot: that feeding -> analysis -> detection
actually fits together, that a planted bursty story is recovered, and that the
output files have the exact shape the evaluation scripts expect.
"""

import os
import sys
import tempfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sskg.config import SSKGConfig
from sskg.data import load_stream
from sskg.pipeline import run


# ---------------------------------------------------------------------------
# a small synthetic corpus with one planted "event"
# ---------------------------------------------------------------------------

QUIET = [
    "قیمت طلا بازار تهران",
    "فیلم سینما جشنواره فجر",
    "تیم فوتبال استقلال بازی",
    "ویتامین تغذیه سلامت بدن",
]
BURST = "آتش سوزی ساختمان پلاسکو تهران آوار"


def make_dataset(path: str, n_windows: int = 4, burst_window: int = 3) -> None:
    """Write an AllData.npy in the real 7-part format.

    Window `burst_window` contains one story that is mentioned twice early on
    and then explodes at the end of the window — which is exactly the shape
    Eq. (8) scores above 1 (Fig. 4).  Every other post is background noise.
    """
    origin = datetime(2017, 1, 1, 0, 0, 0)
    ids, texts, types, dele, dts, users, wins = [], [], [], [], [], [], []
    pid = 1000
    for w in range(n_windows):
        w_ids, w_txt, w_typ, w_del, w_dt, w_usr = [], [], [], [], [], []
        base = origin + timedelta(hours=12 * w)
        is_burst_window = (w + 1) == burst_window
        for i in range(24):
            if is_burst_window and (i < 2 or i >= 12):
                tokens = BURST.split()
            else:
                tokens = QUIET[i % len(QUIET)].split()
            if is_burst_window and i >= 12:
                when = base + timedelta(minutes=690 + (i - 12))   # the burst
            else:
                when = base + timedelta(minutes=i * 25)           # spread out
            w_ids.append(pid)
            w_txt.append(tokens)
            w_typ.append(["PersianWord"] * len(tokens))
            w_del.append("")
            w_dt.append(when)
            w_usr.append(-1)
            pid += 1
        ids.append(w_ids); texts.append(w_txt); types.append(w_typ)
        dele.append(w_del); dts.append(w_dt); users.append(w_usr)
        wins.append(w + 1)
    arr = np.empty(7, dtype=object)
    for k, part in enumerate([ids, texts, types, dele, dts, users, wins]):
        arr[k] = part
    np.save(path, arr, allow_pickle=True)


@pytest.fixture(scope="module")
def workspace():
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        make_dataset("AllData.npy")
        try:
            yield tmp
        finally:
            os.chdir(cwd)


def offline_config(**kw) -> SSKGConfig:
    cfg = SSKGConfig(subject_backend="keyword", kg_backend="heuristic",
                     embed_backend="hashing", verbose=False, **kw)
    return cfg


# ---------------------------------------------------------------------------

def test_dataset_round_trip(workspace):
    windows = load_stream("AllData.npy", window_hours=12)
    assert len(windows) == 4
    assert sum(len(w) for w in windows) == 96
    assert windows[0].posts[0].window == 1
    assert windows[0].posts[0].text.startswith("قیمت طلا")


def test_pipeline_runs_and_writes_both_files(workspace):
    result = run(offline_config())
    assert os.path.exists(result.topics_path)
    assert os.path.exists(result.assignments_path)
    assert result.stats["windows"] == 4
    assert result.stats["posts"] == 96
    assert result.stats["triples_fed"] > 0


def test_knowledge_graphs_are_stored_and_then_reused(workspace):
    """Second run must not rebuild anything — the store is the cache."""
    from sskg.kg_extraction import KnowledgeGraphStore
    store = KnowledgeGraphStore("KnowledgeGraphs")
    assert len(store) > 0
    mtime = os.path.getmtime(store.path)
    run(offline_config())
    assert os.path.getmtime(store.path) == mtime      # untouched on re-run


def test_planted_burst_is_detected(workspace):
    result = run(offline_config())
    df = pd.read_excel(result.topics_path)
    topics = " ".join(str(t) for t in df["Topic"].dropna())
    assert "پلاسکو" in topics
    # the burst lives in window 3, so that window must have produced events
    assert 3 in set(int(x) for x in df["Window Number"])


def test_assignment_file_has_one_sheet_per_window(workspace):
    result = run(offline_config())
    book = pd.ExcelFile(result.assignments_path)
    assert book.sheet_names == [f"Window-{i}" for i in (1, 2, 3, 4)]
    df = book.parse("Window-3")
    assert list(df.columns[:2]) == ["Sequence", "EventNumber"]
    assert len(df) == 24
    assert (df["EventNumber"].astype(str) != "").all()


def test_every_post_appears_exactly_once_in_the_output(workspace):
    result = run(offline_config())
    book = pd.ExcelFile(result.assignments_path)
    seqs = []
    for sheet in book.sheet_names:
        seqs += book.parse(sheet)["Sequence"].tolist()
    assert len(seqs) == 96 and len(set(seqs)) == 96


def test_forgetting_keeps_the_graph_small(workspace):
    """With t_d = 1.2 the stream graph must not grow without bound."""
    strict = run(offline_config(t_d=1.2))
    lax = run(offline_config(t_d=0.0, out_dir="SystemResults_lax"))
    assert strict.stats["final_events"] <= lax.stats["final_events"]


def test_thresh_ts_controls_the_number_of_final_events(workspace):
    # Thresh_ts is a Euclidean distance on un-normalised vectors, so it has no
    # upper bound: a huge value merges everything, a tiny one splits.
    merged = run(offline_config(thresh_ts=1e9, out_dir="SR_merged"))
    split = run(offline_config(thresh_ts=0.05, out_dir="SR_split"))
    assert merged.stats["final_events"] == 1
    assert split.stats["final_events"] >= merged.stats["final_events"]


def test_max_windows_limits_the_run(workspace):
    result = run(offline_config(max_windows=2, out_dir="SR_two"))
    assert result.stats["windows"] == 2
