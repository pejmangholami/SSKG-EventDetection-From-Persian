# -*- coding: utf-8 -*-
"""
SSKG — Section 3.2.2: one knowledge graph per post.

Every post is turned into a set of ordered triples ``[object1, relation,
object2]`` by a large pre-trained language model used in a Few-Shot fashion.
The system prompt below is the one printed in Section 3.2.2 of the paper,
reproduced verbatim.

The extraction is by far the most expensive stage (one LLM call per post), so
this module is built around a **resumable store**:

* triples are appended to ``KnowledgeGraphs/knowledge_graphs.jsonl`` as soon as
  they are produced — killing the process loses at most the in-flight requests;
* on start-up the store is indexed by post id, and only the missing posts are
  sent to the model;
* the pipeline calls :func:`ensure_knowledge_graphs`, which builds whatever is
  missing and then loads everything.  So the first run extracts, and every
  later run just loads — which is exactly the behaviour asked for.

Backends
--------
``openai`` / ``anthropic``   the paper's configuration (an LLM + the Few-Shot
                             prompt).  Requests run in a thread pool with
                             exponential-backoff retries.
``heuristic``                a dependency-free offline extractor: adjacent
                             content-word chunks become entities linked by a
                             generic relation.  It exists so the pipeline can
                             be executed and tested without an API key; it is
                             **not** the paper's configuration and will not
                             reproduce the paper's numbers.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

Triple = Tuple[str, str, str]

# ---------------------------------------------------------------------------
# The paper's Few-Shot prompt (Section 3.2.2, verbatim)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a knowledge graph builder.
Your task is to output the relationships between two objects in the format [object 1, relationship, object 2].
All date-related information must be included.
Example input: John bought a laptop.
Example output: [['John', 'purchased', 'laptop']].
Example input: John built a house in 2019.
Example output: [['John', 'constructed', 'house'], ['house', 'built in', '2019']].
Output Format:
{"Knowledge Graph": "The list of relationships in the format [object 1, relationship, object 2]."}"""


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------

class KnowledgeGraphStore:
    """Append-only JSONL store of per-post knowledge graphs.

    One line per post::

        {"pid": 556790, "triples": [["رب انار", "کاهش", "وزن"], ...],
         "backend": "openai:gpt-4o-mini"}
    """

    def __init__(self, directory: str = "KnowledgeGraphs",
                 filename: str = "knowledge_graphs.jsonl"):
        self.directory = directory
        self.path = os.path.join(directory, filename)
        os.makedirs(directory, exist_ok=True)
        self._graphs: Dict[int, List[Triple]] = {}
        self._lock = threading.Lock()
        self._fh = None
        self.load()

    # -- reading ------------------------------------------------------------
    def load(self) -> int:
        """(Re)build the in-memory index from disk. Returns #posts loaded."""
        self._graphs.clear()
        if not os.path.exists(self.path):
            return 0
        bad = 0
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    self._graphs[int(rec["pid"])] = [tuple(t) for t in rec["triples"]]
                except Exception:
                    bad += 1
        if bad:
            print(f"[kg] warning: skipped {bad} malformed line(s) in {self.path}")
        return len(self._graphs)

    def __contains__(self, pid: int) -> bool:
        return int(pid) in self._graphs

    def __len__(self) -> int:
        return len(self._graphs)

    def get(self, pid: int) -> List[Triple]:
        return self._graphs.get(int(pid), [])

    def missing(self, pids: Iterable[int]) -> List[int]:
        return [int(p) for p in pids if int(p) not in self._graphs]

    # -- writing ------------------------------------------------------------
    def _open(self):
        if self._fh is None:
            self._fh = open(self.path, "a", encoding="utf-8")
        return self._fh

    def put(self, pid: int, triples: Sequence[Triple], backend: str = "") -> None:
        triples = [tuple(map(str, t)) for t in triples if len(t) == 3]
        with self._lock:
            self._graphs[int(pid)] = triples
            fh = self._open()
            fh.write(json.dumps({"pid": int(pid), "triples": triples,
                                 "backend": backend}, ensure_ascii=False) + "\n")
            fh.flush()

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None

    # -- maintenance --------------------------------------------------------
    def compact(self) -> None:
        """Rewrite the file keeping only the last record of each post."""
        self.close()
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            for pid, triples in self._graphs.items():
                fh.write(json.dumps({"pid": pid, "triples": triples},
                                    ensure_ascii=False) + "\n")
        os.replace(tmp, self.path)

    def stats(self) -> str:
        n_tr = sum(len(v) for v in self._graphs.values())
        n = max(1, len(self._graphs))
        return (f"{len(self._graphs)} posts, {n_tr} triples "
                f"({n_tr / n:.1f} per post) in {self.path}")


# ---------------------------------------------------------------------------
# parsing the model answer
# ---------------------------------------------------------------------------

_TRIPLE_RE = re.compile(r"\[([^\[\]]+?)\]")


def parse_triples(answer: str, max_triples: int = 32) -> List[Triple]:
    """Parse an LLM answer into triples, tolerating the usual sloppiness.

    Accepts a JSON object with a ``Knowledge Graph`` key, a bare JSON list, a
    Python-style list of lists, or free text containing ``[a, b, c]`` groups.
    """
    if not answer:
        return []
    text = answer.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()

    # 1) proper JSON
    for candidate in (text,):
        try:
            obj = json.loads(candidate)
        except Exception:
            obj = None
        if obj is not None:
            if isinstance(obj, dict):
                for key in ("Knowledge Graph", "knowledge_graph", "graph",
                            "triples", "relationships"):
                    if key in obj:
                        obj = obj[key]
                        break
            if isinstance(obj, str):
                return parse_triples(obj, max_triples)
            if isinstance(obj, list):
                out = []
                for item in obj:
                    if isinstance(item, (list, tuple)) and len(item) == 3:
                        out.append(tuple(str(x).strip() for x in item))
                if out:
                    return out[:max_triples]

    # 2) fall back to a regex over [a, b, c] groups
    out: List[Triple] = []
    for group in _TRIPLE_RE.findall(text):
        parts = [p.strip().strip("'\"") for p in group.split(",")]
        if len(parts) == 3 and all(parts):
            out.append(tuple(parts))
    return out[:max_triples]


def clean_triples(triples: Sequence[Triple], max_len: int = 60) -> List[Triple]:
    """Drop empty/degenerate triples and normalise whitespace."""
    out: List[Triple] = []
    seen = set()
    for t in triples:
        if len(t) != 3:
            continue
        h, r, o = (re.sub(r"\s+", " ", str(x)).strip() for x in t)
        if not h or not o or h == o:
            continue
        if len(h) > max_len or len(o) > max_len or len(r) > max_len:
            continue
        r = r or "مرتبط با"
        key = (h, r, o)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------

class HeuristicExtractor:
    """Offline stand-in: chunk the post and chain the chunks together.

    The corpus is already tokenised and stop-word free, so consecutive content
    words are grouped into 1–2 token entities and consecutive entities are
    linked by a generic relation.  This produces graphs with the same *shape*
    as Fig. A.10–A.16 (connected chains, sometimes disconnected), which is
    enough to exercise and test the rest of the pipeline.
    """

    name = "heuristic"

    def __init__(self, chunk: int = 2, max_triples: int = 32,
                 relation: str = "مرتبط با"):
        self.chunk = chunk
        self.max_triples = max_triples
        self.relation = relation

    def __call__(self, pid: int, text: str) -> List[Triple]:
        tokens = [t for t in text.split() if t]
        if len(tokens) < 2:
            return []
        entities = [" ".join(tokens[i:i + self.chunk])
                    for i in range(0, len(tokens), self.chunk)]
        entities = [e for e in entities if e]
        triples = [(entities[i], self.relation, entities[i + 1])
                   for i in range(len(entities) - 1)]
        return clean_triples(triples)[:self.max_triples]


class OpenAIExtractor:
    """The paper's configuration, served by the OpenAI chat-completions API."""

    def __init__(self, model: str = "gpt-4o-mini",
                 temperature: Optional[float] = None,
                 max_retries: int = 5, max_triples: int = 32,
                 api_key: Optional[str] = None):
        try:
            from openai import OpenAI
        except ImportError as exc:                            # pragma: no cover
            raise ImportError("pip install openai   (or use --kg-backend heuristic)") from exc
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.max_triples = max_triples
        self.name = f"openai:{model}"

    def __call__(self, pid: int, text: str) -> List[Triple]:
        last = None
        for attempt in range(self.max_retries):
            try:
                kwargs = {}
                if self.temperature is not None:   # otherwise: the API default
                    kwargs["temperature"] = self.temperature
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT},
                              {"role": "user", "content": text}],
                    **kwargs,
                )
                answer = resp.choices[0].message.content
                return clean_triples(parse_triples(answer, self.max_triples))
            except Exception as exc:                          # pragma: no cover
                last = exc
                time.sleep(min(2 ** attempt, 30))
        print(f"[kg] post {pid}: giving up after {self.max_retries} retries ({last})")
        return []


class AnthropicExtractor:
    """Same prompt, served by the Anthropic messages API."""

    def __init__(self, model: str = "claude-sonnet-4-5",
                 temperature: Optional[float] = None,
                 max_retries: int = 5, max_triples: int = 32,
                 api_key: Optional[str] = None):
        try:
            import anthropic
        except ImportError as exc:                            # pragma: no cover
            raise ImportError("pip install anthropic   (or use --kg-backend heuristic)") from exc
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.max_triples = max_triples
        self.name = f"anthropic:{model}"

    def __call__(self, pid: int, text: str) -> List[Triple]:
        last = None
        for attempt in range(self.max_retries):
            try:
                kwargs = {}
                if self.temperature is not None:   # otherwise: the API default
                    kwargs["temperature"] = self.temperature
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": text}],
                    **kwargs,
                )
                answer = "".join(b.text for b in resp.content if b.type == "text")
                return clean_triples(parse_triples(answer, self.max_triples))
            except Exception as exc:                          # pragma: no cover
                last = exc
                time.sleep(min(2 ** attempt, 30))
        print(f"[kg] post {pid}: giving up after {self.max_retries} retries ({last})")
        return []


def build_extractor(cfg) -> Callable[[int, str], List[Triple]]:
    if cfg.kg_backend == "heuristic":
        return HeuristicExtractor(max_triples=cfg.kg_max_triples_per_post)
    if cfg.kg_backend == "openai":
        return OpenAIExtractor(model=cfg.kg_model, temperature=cfg.kg_temperature,
                               max_retries=cfg.kg_max_retries,
                               max_triples=cfg.kg_max_triples_per_post)
    if cfg.kg_backend == "anthropic":
        return AnthropicExtractor(model=cfg.kg_model, temperature=cfg.kg_temperature,
                                  max_retries=cfg.kg_max_retries,
                                  max_triples=cfg.kg_max_triples_per_post)
    raise ValueError(f"unknown kg_backend: {cfg.kg_backend!r}")


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def ensure_knowledge_graphs(posts, cfg, store: Optional[KnowledgeGraphStore] = None,
                            progress: bool = True) -> KnowledgeGraphStore:
    """Make sure every post has a knowledge graph, then return the store.

    Posts already present in ``KnowledgeGraphs/`` are loaded from disk; the
    rest are extracted now and appended to the store.
    """
    store = store or KnowledgeGraphStore(cfg.kg_dir)
    todo = store.missing(p.pid for p in posts)
    print(f"[kg] store: {store.stats()}")
    if not todo:
        print("[kg] all posts already have a knowledge graph — loading from disk.")
        return store

    print(f"[kg] extracting {len(todo)} missing graph(s) with backend "
          f"'{cfg.kg_backend}' ({cfg.kg_workers} worker(s)) ...")
    extractor = build_extractor(cfg)
    backend_name = getattr(extractor, "name", cfg.kg_backend)
    by_id = {p.pid: p for p in posts}

    def work(pid: int):
        triples = extractor(pid, by_id[pid].text)
        store.put(pid, triples, backend_name)
        return pid

    done = 0
    t0 = time.time()
    if cfg.kg_workers > 1 and cfg.kg_backend != "heuristic":
        with ThreadPoolExecutor(max_workers=cfg.kg_workers) as pool:
            for _ in pool.map(work, todo):
                done += 1
                if progress and done % 100 == 0:
                    rate = done / max(1e-9, time.time() - t0)
                    print(f"[kg]   {done}/{len(todo)}  ({rate:.1f} posts/s)")
    else:
        for pid in todo:
            work(pid)
            done += 1
            if progress and done % 1000 == 0:
                print(f"[kg]   {done}/{len(todo)}")

    store.close()
    print(f"[kg] done in {time.time() - t0:.1f}s — {store.stats()}")
    return store
