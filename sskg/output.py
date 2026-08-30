# -*- coding: utf-8 -*-
"""
SSKG — result files.

Two files are written per run, in exactly the format the evaluation scripts of
this project family expect (Section 4.2 of the paper):

``Topic_Systemresult_<stamp>.xls``
    one row per detected event: ``Window Number`` and ``Topic`` (the event's
    title strings joined by " | ").  Feeds the **topic recall** metric.

``ResultsToCompaire_<stamp>.xls``
    one sheet per window, named ``Window-<n>``, with the post id and the event
    number(s) assigned to it (``0`` = no event).  Feeds the **entropy** metric.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    import xlwt
except ImportError:                                          # pragma: no cover
    xlwt = None


def _require_xlwt():
    if xlwt is None:
        raise ImportError(
            "Writing .xls needs the 'xlwt' package (pip install xlwt). "
            "The evaluation scripts read .xls, so it is required."
        )


def stamped_path(out_dir: str, base: str, stamp: str, ext: str = ".xls") -> str:
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{base}_{stamp}{ext}")


def save_topics(path: str, rows: Sequence[Tuple[int, Sequence[str]]]) -> str:
    """`rows` is a sequence of (window_number, [title, title, ...])."""
    _require_xlwt()
    book = xlwt.Workbook(encoding="utf-8")
    sheet = book.add_sheet("Topic_Systemresult")
    sheet.write(0, 0, "Window Number")
    sheet.write(0, 1, "Topic")
    r = 1
    for window, titles in rows:
        titles = [t for t in titles if t]
        if not titles:
            sheet.write(r, 0, int(window))
            sheet.write(r, 1, "")
            r += 1
            continue
        sheet.write(r, 0, int(window))
        sheet.write(r, 1, " | ".join(titles))
        r += 1
    book.save(path)
    return path


def save_post_assignments(path: str,
                          per_window: Sequence[Tuple[int, Sequence[Tuple[int, Sequence[int]]]]]
                          ) -> str:
    """`per_window` is [(window_number, [(post_id, [event numbers]), ...]), ...]."""
    _require_xlwt()
    book = xlwt.Workbook(encoding="utf-8")
    for window, assignments in per_window:
        sheet = book.add_sheet(f"Window-{int(window)}")
        sheet.write(0, 0, "Sequence")
        sheet.write(0, 1, "EventNumber")
        for r, (pid, events) in enumerate(assignments, start=1):
            sheet.write(r, 0, int(pid))
            sheet.write(r, 1, ",".join(str(int(e)) for e in events) if events else "0")
    book.save(path)
    return path


def save_events_report(path: str, events: Iterable) -> str:
    """Human-readable dump of the final event list (tab-separated, UTF-8)."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("event\tsubjects\twindows\tposts\ttitles\n")
        for ev in events:
            fh.write("\t".join([
                str(ev.number),
                ",".join(sorted(ev.subjects)),
                ",".join(str(w) for w in sorted(ev.windows)),
                str(len(ev.post_ids)),
                " | ".join(ev.titles[:10]),
            ]) + "\n")
    return path
