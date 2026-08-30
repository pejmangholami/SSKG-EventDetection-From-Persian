# -*- coding: utf-8 -*-
"""
SSKG — Section 3.1: the *subject stream*.

Algorithm 1 of the paper turns one text stream into eight subject streams:

    for post in TextStream:
        L_score <- ParsBERT_S(post)
        for i, score in L_score:
            if score >= Thresh_PS:
                SubjectStream[i].append(post)

A post may land in zero, one or several subject streams.  Posts that reach no
subject are dropped — this is the paper's dynamic noise removal (Fig. 8, row 9
"Deleted Set": 795 of the 10 284 posts).

Backends
--------
``parsbert``  the paper's configuration: the ParsBERT news classifier
              (``HooshvareLab/bert-fa-base-uncased-clf-persiannews``, the
              8-class Persian-News head of [43]).  Its ``id2label`` is mapped
              onto the paper's eight groups through ``LABEL_ALIASES``.
``keyword``   an offline lexicon classifier so the pipeline runs (and is
              testable) with no model download.  Not the paper's configuration.

Scores are cached on disk, because classifying 10 284 posts is the second most
expensive step of the pipeline after the knowledge-graph extraction.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from .config import SUBJECTS, PAPER_UNLABELLED_POSTS
from .embeddings import normalise_phrase

# ---------------------------------------------------------------------------
# Mapping from a HuggingFace classifier's own label names to our eight groups.
# The Persian-News head of ParsBERT uses (some spelling of) exactly these eight
# classes; extra aliases make the mapping robust across checkpoints/versions.
# ---------------------------------------------------------------------------
LABEL_ALIASES: Dict[str, str] = {
    # social
    "social": "social", "اجتماعی": "social", "society": "social",
    # economic
    "economic": "economic", "economy": "economic", "اقتصادی": "economic",
    "business": "economic",
    # international
    "international": "international", "world": "international",
    "بین الملل": "international", "بین‌الملل": "international",
    "خارجی": "international",
    # political
    "political": "political", "politics": "political", "سیاسی": "political",
    # science & technology
    "science technology": "science_tech", "science": "science_tech",
    "technology": "science_tech", "tech": "science_tech",
    "علمی": "science_tech", "علم و فناوری": "science_tech",
    "علمی و فناوری": "science_tech", "science-technology": "science_tech",
    "it": "science_tech", "digital": "science_tech",
    # cultural & art
    "cultural art": "cultural_art", "culture": "cultural_art",
    "art": "cultural_art", "arts": "cultural_art",
    "فرهنگی": "cultural_art", "هنری": "cultural_art",
    "فرهنگی و هنری": "cultural_art", "cultural-art": "cultural_art",
    "culture art": "cultural_art", "literature": "cultural_art",
    # sport
    "sport": "sport", "sports": "sport", "ورزشی": "sport", "ورزش": "sport",
    # medical
    "medical": "medical", "health": "medical", "پزشکی": "medical",
    "سلامت": "medical", "medicine": "medical",
}


def canonical_subject(label: str) -> Optional[str]:
    key = normalise_phrase(str(label)).lower().replace("_", " ").strip()
    if key in LABEL_ALIASES:
        return LABEL_ALIASES[key]
    key2 = key.replace(" ", "")
    for alias, subj in LABEL_ALIASES.items():
        if alias.replace(" ", "") == key2:
            return subj
    return None


# ---------------------------------------------------------------------------
# offline lexicon backend
# ---------------------------------------------------------------------------

KEYWORD_LEXICON: Dict[str, List[str]] = {
    "social": ["مردم", "شهر", "شهرداری", "جامعه", "خانواده", "حادثه", "آتش",
               "پلیس", "تصادف", "زلزله", "کودک", "زنان", "اجتماعی", "شهروند",
               "آموزش", "مدرسه", "دانشگاه", "ترافیک", "محیط", "زیست", "امداد"],
    "economic": ["اقتصاد", "قیمت", "بازار", "دلار", "سکه", "طلا", "بورس",
                 "تورم", "بانک", "یارانه", "بودجه", "صادرات", "واردات",
                 "خودرو", "مسکن", "اقتصادی", "تولید", "سرمایه", "ارز"],
    "international": ["آمریکا", "ترامپ", "روسیه", "سوریه", "عراق", "ترکیه",
                      "اروپا", "چین", "اسرائیل", "عربستان", "سازمان", "ملل",
                      "جهان", "بین", "الملل", "برجام", "مذاکرات", "خارجه"],
    "political": ["رئیس", "جمهور", "مجلس", "نماینده", "دولت", "انتخابات",
                  "سیاسی", "رهبر", "وزیر", "روحانی", "هاشمی", "اصلاحات",
                  "اصولگرا", "شورا", "قوه", "قضاییه", "نظام", "انقلاب"],
    "science_tech": ["فناوری", "علمی", "تحقیقات", "پژوهش", "اینترنت", "موبایل",
                     "گوشی", "اپلیکیشن", "نرم", "افزار", "هوش", "مصنوعی",
                     "ماهواره", "فضا", "دانشمند", "اختراع", "رباتیک", "سایت"],
    "cultural_art": ["فیلم", "سینما", "بازیگر", "کتاب", "موسیقی", "هنر",
                     "جشنواره", "نمایشگاه", "تئاتر", "شعر", "فرهنگ", "هنری",
                     "کنسرت", "خواننده", "کارگردان", "نویسنده", "رمان"],
    "sport": ["فوتبال", "تیم", "بازی", "استقلال", "پرسپولیس", "لیگ", "گل",
              "مربی", "بازیکن", "ورزش", "قهرمانی", "المپیک", "کشتی", "والیبال",
              "باشگاه", "داور", "مسابقه", "جام"],
    "medical": ["بیماری", "درمان", "پزشک", "سلامت", "دارو", "بیمارستان",
                "سرطان", "ویتامین", "تغذیه", "خون", "قلب", "پوست", "مغز",
                "واکسن", "عفونت", "چربی", "کلسترول", "فشار", "دیابت", "لاغری"],
}


class KeywordSubjectClassifier:
    """Offline lexicon classifier — returns a score in [0, 1] per subject."""

    name = "keyword-lexicon"

    def __init__(self, lexicon: Optional[Dict[str, List[str]]] = None):
        self.lexicon = lexicon or KEYWORD_LEXICON

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        scores = np.zeros((len(texts), len(SUBJECTS)), dtype=np.float32)
        for r, text in enumerate(texts):
            tokens = set(normalise_phrase(text).split())
            if not tokens:
                continue
            for c, subject in enumerate(SUBJECTS):
                hits = sum(1 for kw in self.lexicon[subject] if kw in tokens)
                scores[r, c] = hits
            total = scores[r].sum()
            if total > 0:
                scores[r] /= total          # normalise to a distribution
        return scores


# ---------------------------------------------------------------------------
# ParsBERT backend (the paper's configuration)
# ---------------------------------------------------------------------------

class ParsBertSubjectClassifier:
    """ParsBERT text-classification head, scored over the eight subject groups.

    ``score_mode='softmax'`` follows the way the Persian-News head was trained
    (one distribution over the classes); ``score_mode='sigmoid'`` reads the
    logits as independent per-class probabilities, which is what allows a post
    to reach several subjects at once when Thresh_PS >= 0.5.
    """

    def __init__(self,
                 model_name: str = "HooshvareLab/bert-fa-base-uncased-clf-persiannews",
                 batch_size: int = 32,
                 max_length: int = 128,
                 score_mode: str = "softmax",
                 device: Optional[str] = None,
                 local_files_only: bool = False):
        try:
            import torch                                     # noqa: F401
            from transformers import (AutoModelForSequenceClassification,
                                      AutoTokenizer)
        except ImportError as exc:                           # pragma: no cover
            raise ImportError(
                "The 'parsbert' subject backend needs torch + transformers:\n"
                "    pip install torch transformers\n"
                "Use --subject-backend keyword to run fully offline."
            ) from exc
        import torch

        self.torch = torch
        self.name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        if score_mode not in ("softmax", "sigmoid"):
            raise ValueError(f"unknown score_mode: {score_mode!r}")
        self.score_mode = score_mode
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, local_files_only=local_files_only)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, local_files_only=local_files_only)
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # map the checkpoint's own labels onto the paper's eight subjects
        id2label = self.model.config.id2label
        self.column_of = {}
        unmapped = []
        for idx, label in id2label.items():
            subject = canonical_subject(label)
            if subject is None:
                unmapped.append(label)
            else:
                self.column_of[int(idx)] = SUBJECTS.index(subject)
        if unmapped:
            print(f"[subject] WARNING: checkpoint labels not mapped to a "
                  f"subject group and therefore ignored: {unmapped}. "
                  f"Extend LABEL_ALIASES in sskg/subject_stream.py.")

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        torch = self.torch
        out = np.zeros((len(texts), len(SUBJECTS)), dtype=np.float32)
        for start in range(0, len(texts), self.batch_size):
            chunk = list(texts[start:start + self.batch_size])
            batch = self.tokenizer(chunk, return_tensors="pt", padding=True,
                                   truncation=True,
                                   max_length=self.max_length).to(self.device)
            with torch.no_grad():
                logits = self.model(**batch).logits
            if self.score_mode == "sigmoid":
                probs = torch.sigmoid(logits).float().cpu().numpy()
            else:
                probs = torch.softmax(logits, dim=-1).float().cpu().numpy()
            for r in range(probs.shape[0]):
                for model_col, subject_col in self.column_of.items():
                    out[start + r, subject_col] = max(
                        out[start + r, subject_col], probs[r, model_col])
        return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def build_subject_classifier(cfg):
    if cfg.subject_backend == "keyword":
        return KeywordSubjectClassifier()
    if cfg.subject_backend == "parsbert":
        return ParsBertSubjectClassifier(
            model_name=cfg.subject_model,
            score_mode=getattr(cfg, "subject_score_mode", "softmax"))
    raise ValueError(f"unknown subject_backend: {cfg.subject_backend!r}")


def _cache_path(cfg) -> str:
    mode = getattr(cfg, "subject_score_mode", "softmax")
    tag = f"{cfg.subject_backend}_{cfg.subject_model.replace('/', '_')}_{mode}"
    return os.path.join("cache", f"subject_scores_{tag}")


def compute_subject_scores(posts, cfg, use_cache: bool = True) -> np.ndarray:
    """Return an (n_posts, 8) score matrix, cached on disk."""
    path = _cache_path(cfg)
    ids = [p.pid for p in posts]
    if use_cache and os.path.exists(path + ".npy") and os.path.exists(path + ".json"):
        try:
            with open(path + ".json", encoding="utf-8") as fh:
                meta = json.load(fh)
            scores = np.load(path + ".npy")
            cached = {pid: i for i, pid in enumerate(meta["ids"])}
            if all(pid in cached for pid in ids):
                print(f"[subject] loaded cached scores from {path}.npy")
                return scores[[cached[pid] for pid in ids]]
        except Exception:
            pass

    clf = build_subject_classifier(cfg)
    texts = [p.text for p in posts]
    print(f"[subject] scoring {len(texts)} posts with {getattr(clf, 'name', cfg.subject_backend)} ...")
    scores = clf.predict(texts)

    if use_cache:
        os.makedirs("cache", exist_ok=True)
        np.save(path + ".npy", scores)
        with open(path + ".json", "w", encoding="utf-8") as fh:
            json.dump({"ids": ids, "backend": cfg.subject_backend}, fh)
    return scores


def assign_subjects(scores: np.ndarray, thresh_ps: float) -> List[List[int]]:
    """Algorithm 1: every subject whose score >= Thresh_PS gets the post."""
    return [list(np.nonzero(row >= thresh_ps)[0]) for row in scores]


def calibrate_thresh_ps(scores: np.ndarray,
                        target_unlabelled: int = PAPER_UNLABELLED_POSTS) -> float:
    """Pick Thresh_PS so that `target_unlabelled` posts reach no subject.

    The paper does not report Thresh_PS, but Fig. 8 does report how many posts
    end up with no subject label (795 of 10 284).  Sorting the per-post maximum
    score and cutting at that quantile recovers a threshold consistent with the
    published statistics.
    """
    best = scores.max(axis=1)
    n = len(best)
    target = min(max(int(target_unlabelled), 0), n)
    if target == 0:
        return float(best.min())          # everybody keeps at least one subject
    order = np.sort(best)
    lo = float(order[target - 1])
    hi = float(order[target]) if target < n else lo + 1.0
    # any threshold in (lo, hi] leaves exactly `target` posts unlabelled
    return (lo + hi) / 2.0 if hi > lo else lo + 1e-9


def subject_statistics(assignments: List[List[int]]) -> Dict[str, int]:
    counts = {s: 0 for s in SUBJECTS}
    unlabelled = 0
    for subs in assignments:
        if not subs:
            unlabelled += 1
        for i in subs:
            counts[SUBJECTS[i]] += 1
    counts["(unlabelled)"] = unlabelled
    return counts
