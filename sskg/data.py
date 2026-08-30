# -*- coding: utf-8 -*-
"""
SSKG — dataset loading.

`AllData.npy` is the pre-tokenised Persian Telegram corpus of [47]
(Ranjbar-Khadivi et al., Sep_TD_Tel01, doi:10.17632/372RNWF9PC.1) exactly as
shipped with the PLOT reference implementation.  It is a 7-part object array;
part *p* is a list with one entry per time window:

    [0] post ids          list[list[int]]
    [1] tokenised posts   list[list[list[str]]]
    [2] token types       list[list[list[str]]]   PersianWord/CompoundWord/...
    [3] "deleted" flag    list[list[str|int]]     '' for a normal post
    [4] timestamps        list[list[datetime]]
    [5] channel/user id   list[list[int]]
    [6] window number     list[int]               (a scalar per window)

The file ships 1-hour windows; the gold standard annotates 12-hour windows, so
`load_stream()` re-bins the posts into gold-aligned windows whose numbers are
identical to the gold standard's:

    win = floor((t - midnight_of_first_day) / window_hours) + 1
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterator, List, Optional

import numpy as np


@dataclass(slots=True)
class Post:
    """A single message of the stream."""
    pid: int                 # post id (unique across the corpus)
    tokens: List[str]        # pre-tokenised content words
    types: List[str]         # token types, parallel to `tokens`
    dt: datetime             # publication time
    user: int                # channel id
    window: int              # gold-aligned window number (1-based)
    deleted: object = ""     # the corpus' "deleted" flag, kept verbatim

    @property
    def text(self) -> str:
        return " ".join(self.tokens)

    @property
    def timestamp(self) -> float:
        return self.dt.timestamp()


@dataclass
class Window:
    """All posts published inside one time window."""
    number: int
    posts: List[Post]

    def __len__(self) -> int:
        return len(self.posts)

    @property
    def start(self) -> datetime:
        return min(p.dt for p in self.posts)

    @property
    def end(self) -> datetime:
        return max(p.dt for p in self.posts)


def load_stream(path: str = "AllData.npy",
                window_hours: int = 12,
                max_windows: Optional[int] = None) -> List[Window]:
    """Load `AllData.npy` and return the stream as a list of `Window`s.

    Posts are ordered chronologically inside each window and the windows are
    ordered chronologically, which is what a *stream* means for SSKG: the
    pipeline is fed strictly in publication order.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Copy AllData.npy next to run_sskg.py "
            f"(it ships with the dataset / with the PLOT repository)."
        )
    raw = np.load(path, allow_pickle=True)

    posts: List[Post] = []
    n_win_file = len(raw[0])
    for w in range(n_win_file):
        for j in range(len(raw[0][w])):
            posts.append(
                Post(
                    pid=int(raw[0][w][j]),
                    tokens=list(raw[1][w][j]),
                    types=list(raw[2][w][j]),
                    dt=raw[4][w][j],
                    user=int(raw[5][w][j]),
                    window=-1,
                    deleted=raw[3][w][j],
                )
            )
    if not posts:
        return []

    posts.sort(key=lambda p: (p.dt, p.pid))

    if window_hours:
        first = posts[0].dt
        origin = datetime(first.year, first.month, first.day)  # midnight, day 1
        step = window_hours * 3600
        for p in posts:
            p.window = int((p.dt - origin).total_seconds() // step) + 1
    else:
        # keep the file's own window numbering
        idx = 0
        for w in range(n_win_file):
            for _ in range(len(raw[0][w])):
                posts[idx].window = int(raw[6][w])
                idx += 1

    windows: Dict[int, List[Post]] = {}
    for p in posts:
        windows.setdefault(p.window, []).append(p)

    out = [Window(number=n, posts=windows[n]) for n in sorted(windows)]
    if max_windows is not None:
        out = out[:max_windows]
    return out


def iter_posts(windows: List[Window]) -> Iterator[Post]:
    for w in windows:
        for p in w.posts:
            yield p


def stream_stats(windows: List[Window]) -> str:
    n_posts = sum(len(w) for w in windows)
    if not windows:
        return "empty stream"
    return (
        f"{n_posts} posts in {len(windows)} windows "
        f"({windows[0].start:%Y-%m-%d %H:%M} .. {windows[-1].end:%Y-%m-%d %H:%M}), "
        f"mean {n_posts / len(windows):.1f} posts/window"
    )
