# -*- coding: utf-8 -*-
"""
PLOT - Entropy metric (the reference clustering-quality measure)
================================================================

This module provides the single `Entropy` function used by evaluate.py.

It is the standard multi-label weighted cluster/class entropy (log base 2)
described in the PLOT paper (Section IV-B-1), called directly by evaluate.py
on in-memory label lists.

Direction of the metric depends on argument order:
    Entropy(Samples, Clusters, Classes)  -> Cluster Entropy
    Entropy(Samples, Classes,  Clusters) -> Class   Entropy
Total Entropy is the average of the two (computed in evaluate.py).

For each multi-label sample a fractional score of 1/|labels| is used, and each
sample is greedily assigned to the most frequent class within its cluster
before the per-cluster entropy is accumulated.
"""

import copy
from math import log


def Entropy(Samples, Evals, Desires):
    """Calculate the (cluster or class) entropy.

    (Evals = Clusters & Desires = Classes)  => Cluster Entropy
    (Evals = Classes  & Desires = Clusters) => Class   Entropy
    The first argument is the list of samples. Returns the total entropy HO.
    """

    HO = 0  # Total entropy
    Classes = dict()   # Desire-label  -> indices of samples carrying it
    Clusters = dict()  # Eval-label    -> indices of samples carrying it

    Scores = list()
    for i in range(len(Samples)):
        # Fractional weight of each (possibly multi-label) sample.
        Scores.append(1 / len(Evals[i]))

        for D in Desires[i]:
            if D in Classes:
                Classes[D].append(i)
            else:
                Classes.update({D: [i]})

        for E in Evals[i]:
            if E in Clusters:
                Clusters[E].append(i)
            else:
                Clusters.update({E: [i]})

    # Per-cluster: resolve multi-label samples, then accumulate entropy.
    for Cluster in Clusters:
        Sum = 0  # total score of this cluster
        H = 0    # entropy of this cluster

        S = dict()
        for Class in Classes:
            S.update({Class: 0})

        SC = dict()  # sample index -> its (copied) class labels
        FC = dict()  # class -> frequency within this cluster

        for Idx in Clusters[Cluster]:
            for c in Desires[Idx]:
                if c in FC:
                    FC[c] += 1
                else:
                    FC.update({c: 1})
            SC.update({Idx: copy.deepcopy(Desires[Idx])})

        # Greedily assign each sample to the most frequent class in the cluster.
        while len(FC) != 0:
            MaxD = max(list(FC.items()), key=lambda x: x[1])
            if MaxD[1] == 0:
                break
            m = MaxD[0]
            for s in SC:
                if m in SC[s]:
                    for i in SC[s]:
                        FC[i] -= 1
                    SC[s] = [m]
            FC.pop(m)

        for Idx in Clusters[Cluster]:
            Sum += Scores[Idx]
            C = SC[Idx][0]
            S[C] += Scores[Idx]

        for s in S:
            if S[s] != 0:
                H += S[s] * log((S[s] / Sum), 2)
        # p < 1 makes log(p) negative, so negate to obtain a positive entropy.
        HO += -1 * H

    N = len(Samples)
    HO = HO / N
    return HO
