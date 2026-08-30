# -*- coding: utf-8 -*-
"""
SSKG — Subject Stream Knowledge Graph.

Reference implementation of

    P. Gholami-Dastgerdi, M.-R. Feizi-Derakhshi, P. Salehpour,
    "SSKG: Subject stream knowledge graph, a new approach for event detection
    from text", Ain Shams Engineering Journal, 2024.
    https://doi.org/10.1016/j.asej.2024.103040

Module map (paper section in brackets):

    config          all parameters, with the paper's defaults
    data            AllData.npy loading + gold-aligned re-windowing
    subject_stream  the subject stream                       [3.1, Alg. 1]
    kg_extraction   one knowledge graph per post             [3.2.2]
    embeddings      ParsBERT / offline phrase embeddings     [3.2.3]
    stream_graph    the stream graph                         [3.2.3, Alg. 2-3]
    scoring         NAP / NAF / node score / forgetting      [3.3, Eq. 1-8]
    detection       clustering, titles, label mapping        [3.4, Alg. 4]
    pipeline        feeding -> analysis -> detection          [Fig. 1]
    output          result files for the evaluation scripts  [4.2]
"""

__version__ = "1.0.0"

from .config import SSKGConfig, SUBJECTS          # noqa: F401
