# -*- coding: utf-8 -*-
"""
SSKG — Section 3.3: node scoring and the forgetting mechanism.

    nap_x[i] = number of occurrences of entity x in the i-th TS-second
               interval, from its first occurrence t1 to now          (Eq. 2-4)
    NAP      = [nap_1, ..., nap_n]                                    (Eq. 1)
    naf_i    = nap_i / NodeLife                                       (Eq. 5)
    S_i      = Σ(naf_i · weight_i) / Σ(weight_i)                      (Eq. 6)
    SR_i     = Σ(nafr_i · weight_i) / Σ(weight_i)                     (Eq. 7)
    Score_i  = S_i / SR_i                                             (Eq. 8)

with ``weight`` rising linearly from 0.1 (oldest interval) to 1.0 (newest one),
Fig. 6, and ``TS = 60 s``.

What is ``nafr``?
-----------------
The paper calls ``nafr`` "the inverse frequency" and says the SR average
"essentially reverses the previous approach, assigning higher scores to older
nodes".  Reversing the *frequency vector in time* is the reading that makes the
whole of Table 1 true:

* a perfectly uniform history gives ``naf == naf[::-1]`` hence ``Score = 1`` —
  exactly the paper's "the node is on the verge of bursting but may inherently
  have a uniform frequency";
* a node that was busy in the past and is quiet now gives ``Score < 1``
  ("not bursty, not a candidate");
* a node that is busy now gives ``Score > 1`` ("bursty, candidate for an
  event"), and ``Score -> ∞`` for a node with no past at all.

The element-wise reciprocal reading (``1/naf``) breaks all three (it is also
undefined for the many zero intervals), so this implementation uses the
time-reversal.  ``nafr_mode`` lets you switch to the reciprocal if you want to
compare.

Zero-padding to *now* (Fig. 5) is what distinguishes "was frequent long ago"
from "is frequent right now", and it is applied to every entity, so all NAP
vectors share the same end time.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


def nap_vector(times: Sequence[float], now: float, ts_seconds: int = 60) -> np.ndarray:
    """Eq. (2)-(4) + Fig. 5: occurrence counts per TS-second interval, padded to `now`."""
    if len(times) == 0:
        return np.zeros(1, dtype=np.float64)
    t1 = float(min(times))
    span = max(0.0, float(now) - t1)
    m = max(1, int(np.ceil(span / float(ts_seconds))))
    idx = np.floor((np.asarray(times, dtype=np.float64) - t1) / float(ts_seconds))
    idx = np.clip(idx, 0, m - 1).astype(np.int64)
    return np.bincount(idx, minlength=m).astype(np.float64)


def weight_vector(m: int, wmin: float = 0.1, wmax: float = 1.0) -> np.ndarray:
    """Fig. 6: linear ramp from `wmin` (oldest interval) to `wmax` (newest)."""
    if m <= 1:
        return np.array([wmax], dtype=np.float64)
    return np.linspace(wmin, wmax, m, dtype=np.float64)


def entity_score(times: Sequence[float], now: float, ts_seconds: int = 60,
                 wmin: float = 0.1, wmax: float = 1.0,
                 node_life: Optional[float] = None,
                 nafr_mode: str = "reverse") -> float:
    """Score_i of Eq. (8) for one entity (a node or an edge).

    ``node_life`` (Eq. 5) cancels out in the ratio S/SR, so it only matters if
    you want the intermediate S and SR values; it is accepted for clarity.
    """
    times = [t for t in times if t is not None]
    if not times:
        return 0.0
    nap = nap_vector(times, now, ts_seconds)
    m = len(nap)
    if node_life is None:
        node_life = max(1.0, float(now) - float(min(times)))
    naf = nap / float(node_life)                                    # Eq. (5)
    w = weight_vector(m, wmin, wmax)                                # Fig. 6
    if nafr_mode == "reverse":
        nafr = naf[::-1]
    elif nafr_mode == "reciprocal":
        with np.errstate(divide="ignore"):
            nafr = np.where(naf > 0, 1.0 / np.maximum(naf, 1e-12), 0.0)
    else:
        raise ValueError(f"unknown nafr_mode: {nafr_mode!r}")
    denom = float(w.sum())
    s = float((naf * w).sum()) / denom                              # Eq. (6)
    s_r = float((nafr * w).sum()) / denom                           # Eq. (7)
    if s_r <= 0.0:
        return float("inf") if s > 0.0 else 0.0
    return s / s_r                                                  # Eq. (8)


def score_all(times_list: Sequence[Sequence[float]], now: float,
              ts_seconds: int = 60, wmin: float = 0.1, wmax: float = 1.0,
              nafr_mode: str = "reverse") -> np.ndarray:
    """Vector of Score_i for every entity of an :class:`EntityIndex`."""
    return np.array(
        [entity_score(t, now, ts_seconds, wmin, wmax, nafr_mode=nafr_mode)
         for t in times_list],
        dtype=np.float64,
    )


def interpret(score: float) -> str:
    """Table 1 of the paper."""
    if score < 1.0:
        return "not bursty — not a candidate for an event"
    if score == 1.0:
        return "on the verge of bursting / inherently uniform frequency"
    return "bursty — candidate for an event"


def survivors(scores: np.ndarray, t_d: float) -> List[int]:
    """Indices kept by the forgetting mechanism (Score >= t_d)."""
    return [int(i) for i in np.nonzero(scores >= float(t_d))[0]]
