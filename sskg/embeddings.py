# -*- coding: utf-8 -*-
"""
SSKG — phrase embeddings.

Section 3.2.3 of the paper embeds every node phrase and every edge phrase with
ParsBERT [43] and merges the ones whose cosine distance is small (Algorithm 3).
The same vectors are reused for the event titles (Section 3.4.1) and for the
cross-subject label mapping (Algorithm 4).

Two backends:

* ``parsbert`` — mean-pooled last hidden state of ``HooshvareLab/bert-fa-*``.
  This is what the paper uses.
* ``hashing``  — a deterministic character n-gram hashing embedder.  No model
  download, no torch, milliseconds per phrase.  It makes the pipeline runnable
  and testable offline; it is **not** the paper's configuration.

Speed
-----
Phrases repeat massively in a stream (the same entity appears in hundreds of
posts), so every embedder is wrapped in a two-level cache:

* an in-memory ``dict``  (phrase -> vector)
* a disk cache (``cache/embeddings.{npy,json}``) that survives between runs

and all misses are encoded in **batches**.  All returned vectors are L2
normalised, which turns "cosine distance" into ``1 - dot`` and lets the stream
graph do nearest-neighbour search with one matrix-vector product.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_ARABIC_TO_PERSIAN = {ord("ي"): "ی", ord("ك"): "ک", ord("\u200c"): " "}


def normalise_phrase(text: str) -> str:
    """Light Persian normalisation: unify Arabic/Persian letters, squeeze space."""
    if not text:
        return ""
    text = str(text).translate(_ARABIC_TO_PERSIAN)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def l2_normalise(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim == 1:
        n = float(np.linalg.norm(mat))
        return mat / n if n > 0 else mat
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


# ---------------------------------------------------------------------------
# base class + cache
# ---------------------------------------------------------------------------

class BaseEmbedder:
    """Cached, batched phrase embedder. Sub-classes implement `_encode_batch`."""

    dim: int = 0
    name: str = "base"

    def __init__(self, cache_path: Optional[str] = None, batch_size: int = 64):
        self.batch_size = batch_size
        self._cache: Dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self._cache_path = cache_path
        self._dirty = False
        if cache_path:
            self._load_cache()

    # -- public API ---------------------------------------------------------
    def encode(self, phrases: Sequence[str]) -> np.ndarray:
        """Return an (n, dim) float32 matrix of L2-normalised embeddings."""
        phrases = [normalise_phrase(p) for p in phrases]
        missing = [p for p in dict.fromkeys(phrases) if p not in self._cache]
        for start in range(0, len(missing), self.batch_size):
            chunk = missing[start:start + self.batch_size]
            vecs = l2_normalise(self._encode_batch(chunk))
            with self._lock:
                for phrase, vec in zip(chunk, vecs):
                    self._cache[phrase] = vec.astype(np.float32)
                self._dirty = True
        if not phrases:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._cache[p] for p in phrases])

    def encode_one(self, phrase: str) -> np.ndarray:
        return self.encode([phrase])[0]

    # -- persistence --------------------------------------------------------
    def _load_cache(self) -> None:
        vec_file = self._cache_path + ".npy"
        key_file = self._cache_path + ".json"
        if os.path.exists(vec_file) and os.path.exists(key_file):
            try:
                mat = np.load(vec_file)
                with open(key_file, "r", encoding="utf-8") as fh:
                    meta = json.load(fh)
                if meta.get("model") == self.name and mat.shape[0] == len(meta["keys"]):
                    self._cache = {k: mat[i] for i, k in enumerate(meta["keys"])}
            except Exception:                      # a broken cache is not fatal
                self._cache = {}

    def save_cache(self) -> None:
        if not self._cache_path or not self._dirty or not self._cache:
            return
        os.makedirs(os.path.dirname(self._cache_path) or ".", exist_ok=True)
        keys = list(self._cache.keys())
        mat = np.stack([self._cache[k] for k in keys])
        tmp_npy = self._cache_path + ".tmp.npy"              # atomic-ish write
        np.save(tmp_npy, mat)
        os.replace(tmp_npy, self._cache_path + ".npy")
        with open(self._cache_path + ".json.tmp", "w", encoding="utf-8") as fh:
            json.dump({"model": self.name, "keys": keys}, fh, ensure_ascii=False)
        os.replace(self._cache_path + ".json.tmp", self._cache_path + ".json")
        self._dirty = False

    # -- to implement -------------------------------------------------------
    def _encode_batch(self, phrases: List[str]) -> np.ndarray:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# offline hashing embedder
# ---------------------------------------------------------------------------

class HashingEmbedder(BaseEmbedder):
    """Deterministic character n-gram hashing embedder (offline, no downloads).

    Phrases that share character n-grams land close to each other, which is
    enough to exercise every code path of the pipeline (and to run it at all on
    a machine without the language model).  It is a *stand-in*, not ParsBERT.
    """

    def __init__(self, dim: int = 512, ngram_range=(2, 4), cache_path=None, batch_size=256):
        self.dim = int(dim)
        self.ngram_range = ngram_range
        self.name = f"hashing-{dim}-{ngram_range[0]}{ngram_range[1]}"
        super().__init__(cache_path=cache_path, batch_size=batch_size)

    def _encode_batch(self, phrases: List[str]) -> np.ndarray:
        out = np.zeros((len(phrases), self.dim), dtype=np.float32)
        lo, hi = self.ngram_range
        for row, phrase in enumerate(phrases):
            padded = f" {phrase} "
            for n in range(lo, hi + 1):
                for i in range(max(0, len(padded) - n + 1)):
                    gram = padded[i:i + n]
                    h = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
                    idx = int.from_bytes(h[:4], "little") % self.dim
                    sign = 1.0 if h[4] & 1 else -1.0
                    out[row, idx] += sign
        return out


# ---------------------------------------------------------------------------
# ParsBERT embedder (the paper's configuration)
# ---------------------------------------------------------------------------

class ParsBertEmbedder(BaseEmbedder):
    """Mean-pooled contextual embedding from a HuggingFace BERT checkpoint."""

    def __init__(self,
                 model_name: str = "HooshvareLab/bert-fa-base-uncased",
                 batch_size: int = 64,
                 max_length: int = 32,
                 cache_path: Optional[str] = None,
                 device: Optional[str] = None,
                 local_files_only: bool = False):
        try:
            import torch                                    # noqa: F401
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:                          # pragma: no cover
            raise ImportError(
                "The 'parsbert' embedder needs torch + transformers:\n"
                "    pip install torch transformers\n"
                "Use --embed-backend hashing to run fully offline."
            ) from exc
        import torch

        self.torch = torch
        self.name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, local_files_only=local_files_only)
        self.model = AutoModel.from_pretrained(
            model_name, local_files_only=local_files_only)
        self.model.eval()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(device)
        self.dim = int(self.model.config.hidden_size)
        super().__init__(cache_path=cache_path, batch_size=batch_size)

    def _encode_batch(self, phrases: List[str]) -> np.ndarray:
        torch = self.torch
        batch = self.tokenizer(
            phrases, return_tensors="pt", padding=True,
            truncation=True, max_length=self.max_length,
        ).to(self.device)
        with torch.no_grad():
            hidden = self.model(**batch).last_hidden_state      # (b, t, d)
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return pooled.float().cpu().numpy()


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------

def build_embedder(cfg) -> BaseEmbedder:
    """Create the embedder described by an :class:`SSKGConfig`."""
    cache = cfg.embed_cache
    if cfg.embed_backend == "hashing":
        return HashingEmbedder(dim=cfg.embed_dim_hashing,
                               cache_path=f"{cache}_hashing")
    if cfg.embed_backend == "parsbert":
        tag = cfg.embed_model.replace("/", "_")
        return ParsBertEmbedder(model_name=cfg.embed_model,
                                batch_size=cfg.embed_batch_size,
                                max_length=cfg.embed_max_length,
                                cache_path=f"{cache}_{tag}")
    raise ValueError(f"unknown embed_backend: {cfg.embed_backend!r}")
