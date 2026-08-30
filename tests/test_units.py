# -*- coding: utf-8 -*-
"""Unit tests for the SSKG building blocks (run with: python -m pytest -q)."""

import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sskg.config import SSKGConfig, SUBJECTS
from sskg.detection import FinalEventRegistry, SubjectEvent, cluster_graph, detect_events
from sskg.embeddings import HashingEmbedder, l2_normalise, normalise_phrase
from sskg.kg_extraction import (HeuristicExtractor, KnowledgeGraphStore,
                                clean_triples, parse_triples)
from sskg.scoring import (entity_score, interpret, nap_vector, score_all,
                          survivors, weight_vector)
from sskg.stream_graph import StreamGraph
from sskg.subject_stream import (KeywordSubjectClassifier, assign_subjects,
                                 calibrate_thresh_ps, canonical_subject,
                                 subject_statistics)


# ---------------------------------------------------------------------------
# Section 3.3 — NAP / NAF / node score
# ---------------------------------------------------------------------------

def test_nap_counts_occurrences_per_interval():
    # three occurrences: two in the first minute, one in the third
    times = [0.0, 30.0, 150.0]
    nap = nap_vector(times, now=180.0, ts_seconds=60)
    assert list(nap) == [2.0, 0.0, 1.0]


def test_nap_is_zero_padded_up_to_now():
    """Fig. 5: a node silent for a long time keeps trailing zeros."""
    nap = nap_vector([0.0, 10.0], now=600.0, ts_seconds=60)
    assert len(nap) == 10
    assert nap[0] == 2.0
    assert nap[1:].sum() == 0.0


def test_weight_vector_matches_fig6():
    w = weight_vector(5, 0.1, 1.0)
    assert w[0] == pytest.approx(0.1)
    assert w[-1] == pytest.approx(1.0)
    assert np.all(np.diff(w) > 0)          # older intervals weigh less


def test_score_is_one_for_a_uniform_history():
    """Table 1: a uniform frequency sits exactly on Score = 1."""
    times = [i * 60.0 for i in range(20)]
    assert entity_score(times, now=20 * 60.0) == pytest.approx(1.0, abs=1e-9)


def test_score_above_one_for_a_bursty_entity():
    """Fig. 4: quiet for a long time, then a burst -> Score > 1."""
    times = [0.0, 60.0] + [3000.0 + i for i in range(40)]
    assert entity_score(times, now=3100.0) > 1.0


def test_score_below_one_for_a_stale_entity():
    """Busy in the past, silent now -> Score < 1 -> forgotten."""
    times = [float(i) for i in range(40)]
    assert entity_score(times, now=5000.0) < 1.0


def test_score_is_scale_invariant_in_node_life():
    times = [0.0, 60.0, 120.0]
    a = entity_score(times, now=180.0, node_life=180.0)
    b = entity_score(times, now=180.0, node_life=999.0)
    assert a == pytest.approx(b)           # NodeLife cancels in S / SR


def test_interpretation_table1():
    assert "not bursty" in interpret(0.5)
    assert "verge" in interpret(1.0)
    assert "bursty" in interpret(3.0)


def test_survivors_applies_td():
    scores = np.array([0.5, 1.0, 1.3, 9.0])
    assert survivors(scores, 1.2) == [2, 3]


def test_score_all_matches_entity_score():
    times = [[0.0, 60.0], [0.0]]
    got = score_all(times, now=120.0)
    assert got[0] == pytest.approx(entity_score(times[0], 120.0))


# ---------------------------------------------------------------------------
# embeddings
# ---------------------------------------------------------------------------

def test_normalise_phrase_unifies_arabic_letters():
    assert normalise_phrase("كتاب  عربي") == "کتاب عربی"


def test_hashing_embedder_is_deterministic_and_normalised():
    emb = HashingEmbedder(dim=64)
    a = emb.encode(["حادثه پلاسکو", "حادثه پلاسکو", "فوتبال"])
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
    assert np.allclose(a[0], a[1])
    assert float(a[0] @ a[2]) < float(a[0] @ a[1])


def test_embedding_cache_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "emb")
        e1 = HashingEmbedder(dim=32, cache_path=path)
        v1 = e1.encode(["آتش سوزی"])[0]
        e1.save_cache()
        e2 = HashingEmbedder(dim=32, cache_path=path)
        assert "آتش سوزی" in e2._cache
        assert np.allclose(e2.encode(["آتش سوزی"])[0], v1)


# ---------------------------------------------------------------------------
# Section 3.2.2 — knowledge graphs
# ---------------------------------------------------------------------------

def test_parse_triples_accepts_the_paper_output_format():
    answer = '{"Knowledge Graph": "[[\'John\', \'purchased\', \'laptop\']]"}'
    assert parse_triples(answer) == [("John", "purchased", "laptop")]


def test_parse_triples_accepts_bare_lists_and_code_fences():
    answer = "```json\n[[\"a\", \"r\", \"b\"], [\"b\", \"r2\", \"c\"]]\n```"
    assert parse_triples(answer) == [("a", "r", "b"), ("b", "r2", "c")]


def test_parse_triples_falls_back_to_regex():
    assert parse_triples("noise [x, rel, y] more noise") == [("x", "rel", "y")]


def test_clean_triples_drops_degenerate_ones():
    got = clean_triples([("a", "r", "a"), ("", "r", "b"), ("a", "r", "b"),
                         ("a", "r", "b")])
    assert got == [("a", "r", "b")]


def test_heuristic_extractor_builds_a_chain():
    triples = HeuristicExtractor()(1, "رب انار خون چربی کلسترول")
    assert triples and all(len(t) == 3 for t in triples)


def test_kg_store_is_resumable():
    with tempfile.TemporaryDirectory() as tmp:
        store = KnowledgeGraphStore(tmp)
        store.put(7, [("a", "r", "b")], "test")
        store.close()
        again = KnowledgeGraphStore(tmp)
        assert 7 in again
        assert again.get(7) == [("a", "r", "b")]
        assert again.missing([7, 8]) == [8]


# ---------------------------------------------------------------------------
# Section 3.1 — subject stream
# ---------------------------------------------------------------------------

def test_canonical_subject_maps_checkpoint_labels():
    assert canonical_subject("Science Technology") == "science_tech"
    assert canonical_subject("ورزشی") == "sport"
    assert canonical_subject("nonsense") is None


def test_algorithm1_allows_zero_one_or_many_subjects():
    scores = np.array([[0.9, 0.0, 0, 0, 0, 0, 0, 0],
                       [0.2, 0.3, 0, 0, 0, 0, 0, 0],
                       [0.01] * 8])
    got = assign_subjects(scores, 0.15)
    assert got[0] == [0]
    assert got[1] == [0, 1]
    assert got[2] == []                     # dropped: the paper's noise removal


def test_calibrate_thresh_ps_hits_the_target_count():
    rng = np.random.default_rng(0)
    scores = rng.random((100, 8))
    thr = calibrate_thresh_ps(scores, target_unlabelled=10)
    unlabelled = sum(1 for row in assign_subjects(scores, thr) if not row)
    assert unlabelled == 10


def test_keyword_classifier_finds_the_right_subject():
    clf = KeywordSubjectClassifier()
    scores = clf.predict(["تیم فوتبال استقلال بازی لیگ"])
    assert SUBJECTS[int(np.argmax(scores[0]))] == "sport"


def test_subject_statistics_counts_unlabelled():
    st = subject_statistics([[0], [], [0, 3]])
    assert st["(unlabelled)"] == 1
    assert st["social"] == 2


# ---------------------------------------------------------------------------
# Section 3.2.3 — stream graph (Algorithms 2 and 3)
# ---------------------------------------------------------------------------

def _graph(t_n=0.10, t_e=0.10):
    return StreamGraph("test", HashingEmbedder(dim=128), t_n=t_n, t_e=t_e)


def test_feeding_creates_nodes_edges_and_relations():
    g = _graph()
    g.feed([("آتش سوزی", "در", "ساختمان پلاسکو")], when=0.0, post_id=1)
    assert len(g.nodes) == 2 and len(g.edges) == 1 and len(g.relations) == 1


def test_identical_phrases_are_merged_not_duplicated():
    g = _graph()
    g.feed([("پلاسکو", "حادثه", "تهران")], when=0.0, post_id=1)
    g.feed([("پلاسکو", "حادثه", "تهران")], when=10.0, post_id=2)
    assert len(g.nodes) == 2
    assert g.nodes.count[0] == 2
    assert g.nodes.posts[0] == {1, 2}
    assert len(g.nodes.times[0]) == 2       # both occurrences are remembered


def test_merge_threshold_zero_keeps_every_distinct_phrase():
    g = _graph(t_n=0.0, t_e=0.0)
    g.feed([("پلاسکو", "r", "تهران")], 0.0, 1)
    g.feed([("پلاسکوی", "r", "تهرانی")], 1.0, 2)
    assert len(g.nodes) == 4


def test_high_merge_threshold_collapses_similar_phrases():
    g = _graph(t_n=0.9, t_e=0.9)
    g.feed([("پلاسکو", "r", "تهران")], 0.0, 1)
    g.feed([("پلاسکوی", "r2", "تهرانی")], 1.0, 2)
    assert len(g.nodes) == 2                # merged into the existing entities


def test_prune_removes_entities_and_reindexes_relations():
    g = _graph()
    g.feed([("a", "r", "b"), ("c", "r", "d")], 0.0, 1)
    keep_nodes = [0, 1]                     # keep a, b — drop c, d
    g.prune(keep_nodes, list(range(len(g.edges))))
    assert len(g.nodes) == 2
    assert len(g.relations) == 1
    n1, e, n2, _, _ = g.relations[0]
    assert g.nodes.title(n1) == "a" and g.nodes.title(n2) == "b"


def test_prune_keeps_the_exact_string_index_consistent():
    g = _graph()
    g.feed([("a", "r", "b"), ("c", "r", "d")], 0.0, 1)
    g.prune([2, 3], list(range(len(g.edges))))
    g.feed([("c", "r", "d")], 5.0, 2)       # must reuse the surviving entities
    assert len(g.nodes) == 2
    assert g.nodes.count[0] == 2


# ---------------------------------------------------------------------------
# Section 3.4 — detection and Algorithm 4
# ---------------------------------------------------------------------------

def test_clustering_separates_two_disconnected_stories():
    g = _graph()
    g.feed([("پلاسکو", "آتش", "تهران")], 0.0, 1)
    g.feed([("استقلال", "بازی", "پرسپولیس")], 1.0, 2)
    edge_scores = np.ones(len(g.edges))
    clusters = cluster_graph(g, edge_scores, method="louvain")
    assert len(clusters) == 2


def test_detect_events_produces_titles_and_posts():
    g = _graph()
    g.feed([("پلاسکو", "آتش", "تهران")], 0.0, 7)
    node_scores = np.full(len(g.nodes), 2.0)
    edge_scores = np.full(len(g.edges), 2.0)
    events = detect_events(g, node_scores, edge_scores, window=1)
    assert len(events) == 1
    assert events[0].post_ids == {7}
    assert events[0].titles and events[0].keywords


def test_registry_merges_similar_labels_and_splits_far_ones():
    v1 = l2_normalise(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    v2 = l2_normalise(np.array([0.99, 0.14, 0.0], dtype=np.float32))
    v3 = l2_normalise(np.array([0.0, 0.0, 1.0], dtype=np.float32))
    reg = FinalEventRegistry(thresh_ts=0.05)

    def ev(vec, name):
        return SubjectEvent("s", 1, [0], [name], {1}, 1.0, vec)

    assert reg.add(ev(v1, "a")) == 1
    assert reg.add(ev(v2, "b")) == 1        # close enough -> same final event
    assert reg.add(ev(v3, "c")) == 2        # far -> a new final event


def test_registry_with_paper_threshold_merges_everything():
    """Thresh_ts = 2 is the paper's best value; cosine distance <= 2 always."""
    reg = FinalEventRegistry(thresh_ts=2.0)
    rng = np.random.default_rng(0)
    for _ in range(5):
        vec = l2_normalise(rng.normal(size=8).astype(np.float32))
        number = reg.add(SubjectEvent("s", 1, [0], ["t"], {1}, 1.0, vec))
    assert number == 1 and len(reg.events) == 1


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

def test_stamp_is_parseable_by_the_evaluation_script():
    stamp = SSKGConfig().stamp()
    params = dict(item.split("-", 1) for item in stamp.split("_"))
    assert params["td"] == "1.2" and params["clu"] == "louvain"
