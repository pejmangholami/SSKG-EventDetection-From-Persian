# SSKG — Subject Stream Knowledge Graph

> **English** | [فارسی](#فارسی)

Reference implementation of

> **SSKG: Subject stream knowledge graph, a new approach for event detection from text**
> Pejman Gholami-Dastgerdi, Mohammad-Reza Feizi-Derakhshi, Pedram Salehpour
> University of Tabriz — *Ain Shams Engineering Journal* (2024).
> [doi:10.1016/j.asej.2024.103040](https://doi.org/10.1016/j.asej.2024.103040)

Self-contained: read the data, build the knowledge graphs, run the full
pipeline, evaluate against the gold standard. Nothing else is needed.

---

## What SSKG does

SSKG detects events in a Persian Telegram stream by combining two ideas:

1. **Subject stream** — the single text stream is split into eight
   subject-specific streams, so a huge story (the Plasco fire) can no longer
   drown out a small concurrent one, and off-topic posts are dropped
   dynamically.
2. **Stream graph** — a knowledge graph enriched with time. Every post becomes
   a set of triples; the triples of a subject are merged into one graph in
   which each node and edge remembers *when* and *how often* it occurred. An
   event is a cluster of *bursty* entities in that graph.

| Stage | Paper | What happens |
|------:|-------|--------------|
| 1 | §3.1, Alg. 1 | **Subject stream** — ParsBERT labels each post with 0…8 subjects; unlabelled posts are discarded. |
| 2 | §3.2.2 | **Knowledge graph per post** — an LLM extracts `[object, relation, object]` triples with the paper's Few-Shot prompt. |
| 3 | §3.2.3, Alg. 2–3 | **Stream graph** — triples are merged into the subject's graph; similar phrases merge by cosine distance (`t_n`, `t_e`); occurrence times are stored on every node and edge. |
| 4 | §3.3, Eq. 1–8 | **Analysis** — the NAP/NAF burstiness score is computed and entities below `t_d` are forgotten. |
| 5 | §3.4, Alg. 4 | **Detection** — the surviving graph is clustered (Louvain), each cluster is titled and traced back to its posts, and subject labels are merged into global events. |

---

## Install

Python 3.9+.

```bash
pip install -r requirements.txt
```

The first real run downloads ParsBERT (≈ 600 MB) from the HuggingFace Hub and
needs an LLM API key for the knowledge-graph extraction. If you want to try the
pipeline first without either, see [Offline mode](#offline-mode).

---

## Quick start

```bash
# 1. build the per-post knowledge graphs (once; resumable)
export OPENAI_API_KEY=...
python build_knowledge_graphs.py

# 2. run the pipeline (loads the graphs from step 1)
python run_sskg.py

# 3. evaluate against the gold standard
cd evaluation && python evaluate.py
```

Step 1 is optional: `run_sskg.py` builds any missing knowledge graph itself
before it starts. Running it separately just lets you watch (and pay for) that
stage on its own.

### Offline mode

```bash
python run_sskg.py --offline          # ≈ 20 s for the whole corpus
```

`--offline` replaces the three learned components with dependency-free
stand-ins — a keyword classifier, an n-gram triple extractor and a character
hashing embedder — so the whole pipeline runs with no downloads and no API key.
Use it to smoke-test an installation, to develop, and to run the tests.
**It does not reproduce the paper's numbers**; on the real corpus it reaches
Topic Recall ≈ 0.42 instead of the paper's 0.6786, and its "events" are often
channel boilerplate rather than real stories.

---

## The data

`AllData.npy` is the Persian Telegram corpus of [47]
([doi:10.17632/372RNWF9PC.1](https://doi.org/10.17632/372RNWF9PC.1)),
already pre-processed and tokenised — the same file the PLOT repository ships,
read directly in Python with no database. It is a 7-part object array; part *p*
holds one list per time window:

| index | content |
|------:|---------|
| 0 | post ids |
| 1 | tokenised posts |
| 2 | token types (`PersianWord`, `CompoundWord`, `Integer`, `EnglishWord`) |
| 3 | "deleted" flag |
| 4 | timestamps |
| 5 | channel id |
| 6 | window number (one scalar per window) |

10 284 posts, 1–31 January 2017. The file is binned into **1-hour** windows,
but the gold standard annotates **12-hour** windows, so `sskg/data.py` re-bins
the stream into 62 gold-aligned windows whose numbers are identical to the gold
standard's:

```
win = floor((t − midnight of the first day) / window_hours) + 1
```

---

## Parameters

Set on the command line (`python run_sskg.py --help`) or in `sskg/config.py`.

| Parameter | Default | Paper | Meaning |
|-----------|---------|-------|---------|
| `--thresh-ps` | `0.5` | §3.1.2 | `Thresh_PS`, Alg. 1: a post joins every subject scoring at least this. |
| `--subject-score-mode` | `softmax` | — | how the classifier logits become `L_score`; see the note below. |
| `--calibrate-thresh-ps` | off | — | fit `Thresh_PS` so that 795 posts stay unlabelled, as in Fig. 8. |
| `--t-n` | `0.15` | §3.2.3 | cosine-distance threshold for merging **node** phrases (Alg. 3). |
| `--t-e` | `0.20` | §3.2.3 | cosine-distance threshold for merging **edge** phrases (Alg. 3). |
| `--ts-seconds` | `60` | §3.3 | `TS`, the length of the time slot the NAP and NAF vectors are built on. |
| `--t-d` | `1.2` | Table 3 | forgetting threshold on the entity score. |
| `--thresh-ts` | `2.0` | Table 3 | `Thresh_ts`, Alg. 4: **Euclidean** distance between (un-normalised) title embeddings below which two labels become one event. |
| `--clustering` | `louvain` | Table 2 | `louvain`, `spectral` or `hcluster`. |
| `--window-hours` | `12` | §4.1 | detection cadence; 12 keeps the windows gold-aligned. |
| `--forget-every` | `1` | Fig. 1 | run analysis + forgetting every *N* windows ("TimeStep"). |
| `--min-event-size` | `2` | — | clusters smaller than this are not reported. |

Weights rise linearly from **0.1** (oldest interval) to **1.0** (newest), Fig. 6.

A few reading decisions are worth stating explicitly, because the paper leaves
room for interpretation:

* **`nafr` (Eq. 7) is the frequency vector reversed in time**, not its
  element-wise reciprocal. Only the reversal makes all three rows of Table 1
  true at once — in particular a perfectly uniform history then gives exactly
  `Score = 1`, the paper's "on the verge of bursting … uniform frequency". The
  reciprocal is also undefined on the many empty intervals. `sskg/scoring.py`
  implements both (`nafr_mode`), and the reversal is the default.
* **`Thresh_ts = 2` and `t_d = 1.2`.** The caption of Table 3 and the text of
  §4.3 swap these two values; the best cell of Table 3 (0.6786) sits at
  `Thresh_ts = 2`, `t_d = 1.2`, so that is the default here.
* **Algorithm 4 uses the Euclidean distance on un-normalised vectors.** Each
  subject event is represented by its title embedding: the plain mean of the
  raw ParsBERT embeddings of the three members of the title triple, with no L2
  normalisation of the members or of the mean. `Thresh_ts` is compared with the
  Euclidean distance between these vectors. This is *not* the cosine distance
  of Algorithm 3 (`t_n`, `t_e`), which stays cosine on normalised vectors. Raw
  mean-pooled ParsBERT vectors have a norm of roughly 17–20, so Euclidean
  distances are on a much larger scale than cosine distances; at
  `Thresh_ts = 2` in practice only (near-)identical titles are merged.
* **Analysis runs before detection** (Fig. 7): score → forget → cluster, so an
  event is always a cluster of bursty entities.
* **`Thresh_PS = 0.5` and multi-label subjects.** The Persian-News head was
  trained with a softmax, so at most one of the eight groups can score above
  0.5 and every post lands in at most one subject stream. Fig. 8 does report
  small overlaps between the subject sets, which needs the logits read as
  independent per-class probabilities instead: `--subject-score-mode sigmoid`.
  Both readings are implemented; `softmax` is the default because it is how the
  checkpoint was trained.
* **LLM sampling parameters.** Nothing is sent unless you ask for it, so the
  API's own defaults apply (`--kg-temperature` overrides that). The default
  model is `gpt-4o-mini`; set `--kg-model` to whichever GPT you want.

---

## Execution model and caching

The stream is replayed window by window; the three phases of Fig. 1 (feeding,
analysis, detection) run for each window. Everything expensive is cached, so a
second run of the same configuration is dominated by the graph work alone:

| Cache | Location | Rebuilt when |
|-------|----------|--------------|
| Subject scores | `cache/subject_scores_*.npy` | the backend or the post set changes |
| Knowledge graphs | `KnowledgeGraphs/knowledge_graphs.jsonl` | never — append-only, keyed by post id |
| Phrase embeddings (raw, un-normalised) | `cache/embeddings_*.npy` | the embedding model or the cache format changes |

Other speed work, since the printed algorithms are deliberately naive:

* Algorithm 3 re-embeds the whole entity list on every UpSert. Here each phrase
  is embedded **once**, an exact-string fast path skips the search for phrases
  already seen, and the nearest neighbour comes from a single BLAS
  matrix-vector product against a pre-allocated, L2-normalised matrix — merging
  keeps a running mean, so members are never revisited.
* Triples of a post are embedded in one batch, not one call per phrase.
* Knowledge-graph extraction runs in a thread pool (`--kg-workers`) with
  exponential-backoff retries; the store is flushed per post, so an interrupted
  extraction resumes exactly where it stopped.
* Scores are vectorised with NumPy over each entity's occurrence times.

Delete `cache/` if you change the code that produces it; delete
`KnowledgeGraphs/` only if you want to pay for the LLM again.

---

## Output

Written to `SystemResults/`, with the parameter set stamped into the filename:

| File | Contents | Feeds |
|------|----------|-------|
| `Topic_Systemresult_<stamp>.xls` | one row per detected event: window number + title strings joined by `\|` | **Topic Recall** |
| `ResultsToCompaire_<stamp>.xls` | one sheet per window (`Window-<n>`): post id + event number(s), `0` = no event | **Entropy** |
| `FinalEvents_<stamp>.tsv` | human-readable dump of the global event list | reading |

Each event reports both the paper's title (the triple of the stream graph
closest to the cluster label in vector space, §3.4.1) and the surface forms of
its top entities — the evaluation matches titles as substrings of the post
text, and single entity phrases are what actually appear verbatim in posts.

---

## Evaluation

```bash
cd evaluation
python evaluate.py
```

Reads `../SystemResults/`, compares against
`GoldenStandard/GoldenStandard_TopicID_and_TopicString.xlsx` and writes
`evaluation/Final_Evaluation_Report.xlsx` with **Topic Recall** (Eq. 13:
`TP / (TP + FN)`, a gold event counts as detected when one of its words appears
in a system title) and **Class / Cluster / Total Entropy** (Eq. 9–12).

`evaluate.py` and `EvaluateFunctional.py` are the same scripts used by the PLOT
release, unchanged, so numbers from the two papers are directly comparable.

Paper results on this corpus (Table 5): Topic Recall **0.6786**, Class Entropy
**0.3544**, Cluster Entropy 1.6846, Total Entropy 1.0195.

---

## Tests

```bash
python -m pytest tests/ -q          # 44 tests, ~1 s, no network needed
```

`tests/test_units.py` covers the equations and data structures — that a uniform
history scores exactly 1 and a burst scores above 1 (Table 1), that NAP is
zero-padded to *now* (Fig. 5), that merging and forgetting keep the graph
consistent, that Algorithm 1 admits zero/one/many subjects, that the LLM answer
parser survives the usual formats.

`tests/test_pipeline.py` writes a synthetic `AllData.npy` containing one
planted story that is mentioned twice and then bursts, runs the whole pipeline
on it, and checks that the story is recovered, that the store is reused on a
second run, and that both result files have exactly the shape `evaluate.py`
expects.

---

## Folder layout

```
SSKG/
├── run_sskg.py                  ← the full pipeline
├── build_knowledge_graphs.py    ← stage 2 on its own (resumable)
├── requirements.txt
├── AllData.npy                  ← the pre-tokenised corpus
├── sskg/
│   ├── config.py                ← every parameter, with the paper's defaults
│   ├── data.py                  ← loading + gold-aligned re-windowing
│   ├── subject_stream.py        ← §3.1, Algorithm 1
│   ├── kg_extraction.py         ← §3.2.2 + the resumable store
│   ├── embeddings.py            ← §3.2.3, cached phrase embeddings
│   ├── stream_graph.py          ← §3.2.3, Algorithms 2–3
│   ├── scoring.py               ← §3.3, Eq. 1–8, forgetting
│   ├── detection.py             ← §3.4, clustering, titles, Algorithm 4
│   ├── pipeline.py              ← Fig. 1 orchestration
│   └── output.py                ← the .xls result files
├── KnowledgeGraphs/             ← the extracted triples (cache + artefact)
├── SystemResults/               ← pipeline output
├── evaluation/                  ← Topic Recall + Entropy
└── tests/
```

## Citation

```bibtex
@article{GholamiDastgerdi2024SSKG,
  title   = {SSKG: Subject stream knowledge graph, a new approach for event
             detection from text},
  author  = {Gholami-Dastgerdi, Pejman and Feizi-Derakhshi, Mohammad-Reza and
             Salehpour, Pedram},
  journal = {Ain Shams Engineering Journal},
  year    = {2024},
  doi     = {10.1016/j.asej.2024.103040}
}
```

---
---

<a name="فارسی"></a>

# SSKG — گراف دانش جریان موضوعی

> [English](#sskg--subject-stream-knowledge-graph) | **فارسی**

پیاده‌سازی مرجع مقاله‌ی:

> **SSKG: Subject stream knowledge graph, a new approach for event detection from text**
> پژمان غلامی‌دستگردی، محمدرضا فیضی‌درخشی، پدرام صالح‌پور
> دانشگاه تبریز — مجله‌ی *Ain Shams Engineering Journal* (۲۰۲۴)
> [doi:10.1016/j.asej.2024.103040](https://doi.org/10.1016/j.asej.2024.103040)

این مخزن خودکفاست: داده را می‌خوانید، گراف‌های دانش را می‌سازید، کل خط لوله را
اجرا می‌کنید و نتیجه را با استاندارد طلایی ارزیابی می‌کنید.

---

## SSKG چه می‌کند؟

SSKG با ترکیب دو ایده، رخدادها را در جریان پیام‌های فارسی تلگرام تشخیص می‌دهد:

۱. **جریان موضوعی** — جریان متنی واحد به هشت جریان موضوعی شکسته می‌شود تا یک
   رخداد بزرگ (مثل حادثه‌ی پلاسکو) رخدادهای کوچک هم‌زمان را زیر سایه نبرد، و
   داده‌های نامرتبط به‌صورت پویا حذف شوند.

۲. **گراف جریان** — یک گراف دانش غنی‌شده با زمان. هر پست به مجموعه‌ای از
   تریپل‌ها تبدیل می‌شود؛ تریپل‌های هر موضوع در یک گراف ادغام می‌شوند که در آن هر
   گره و یال به‌خاطر می‌سپارد *چه زمانی* و *چند بار* رخ داده است. رخداد، خوشه‌ای
   از موجودیت‌های **منفجرشده (bursty)** در همین گراف است.

| مرحله | مقاله | شرح |
|------:|-------|-----|
| ۱ | §۳٫۱، الگوریتم ۱ | **جریان موضوعی** — ParsBERT به هر پست صفر تا هشت برچسب موضوعی می‌دهد؛ پست بی‌برچسب حذف می‌شود. |
| ۲ | §۳٫۲٫۲ | **گراف دانش هر پست** — یک مدل زبانی بزرگ با همان پرامپت Few-Shot مقاله، تریپل‌های `[شیء، رابطه، شیء]` را استخراج می‌کند. |
| ۳ | §۳٫۲٫۳، الگوریتم‌های ۲ و ۳ | **گراف جریان** — تریپل‌ها در گراف موضوع ادغام می‌شوند؛ عبارت‌های مشابه بر اساس فاصله‌ی کسینوسی (`t_n`، `t_e`) یکی می‌شوند و زمان هر رخداد روی گره/یال ذخیره می‌شود. |
| ۴ | §۳٫۳، روابط ۱ تا ۸ | **تحلیل** — امتیاز انفجار NAP/NAF محاسبه و موجودیت‌های زیر `t_d` فراموش می‌شوند. |
| ۵ | §۳٫۴، الگوریتم ۴ | **تشخیص** — گراف بازمانده خوشه‌بندی (Louvain) می‌شود، هر خوشه عنوان می‌گیرد و به پست‌هایش برمی‌گردد، و برچسب‌های موضوعی در رخدادهای سراسری ادغام می‌شوند. |

---

## نصب

پایتون ۳٫۹ یا بالاتر.

```bash
pip install -r requirements.txt
```

اولین اجرای واقعی، ParsBERT (حدود ۶۰۰ مگابایت) را از HuggingFace دانلود می‌کند
و برای استخراج گراف دانش به کلید API نیاز دارد. اگر می‌خواهید بدون هیچ‌کدام
امتحان کنید، بخش [حالت آفلاین](#حالت-آفلاین) را ببینید.

---

## شروع سریع

```bash
# ۱) ساخت گراف دانش هر پست (یک‌بار؛ قابل ازسرگیری)
export OPENAI_API_KEY=...
python build_knowledge_graphs.py

# ۲) اجرای خط لوله (گراف‌های مرحله‌ی ۱ را لود می‌کند)
python run_sskg.py

# ۳) ارزیابی با استاندارد طلایی
cd evaluation && python evaluate.py
```

مرحله‌ی ۱ اختیاری است: `run_sskg.py` خودش هر گراف دانشِ نبوده را پیش از شروع
می‌سازد و ذخیره می‌کند. اجرای جداگانه فقط برای این است که این مرحله را جدا
ببینید و کنترل کنید.

### حالت آفلاین

```bash
python run_sskg.py --offline          # حدود ۲۰ ثانیه برای کل پیکره
```

سوییچ `--offline` هر سه مؤلفه‌ی یادگیرنده را با جایگزین‌های بدون وابستگی عوض
می‌کند — یک دسته‌بند واژگانی، یک استخراج‌گر تریپل مبتنی بر n-gram و یک امبدر
هشینگ کاراکتری — تا کل خط لوله بدون دانلود و بدون کلید API اجرا شود. برای تست
نصب، توسعه و اجرای تست‌ها از آن استفاده کنید.
**این حالت اعداد مقاله را بازتولید نمی‌کند**؛ روی پیکره‌ی واقعی به Topic Recall
حدود ۰٫۴۲ می‌رسد (به‌جای ۰٫۶۷۸۶ مقاله) و «رخداد»هایش اغلب امضای کانال‌ها هستند
نه خبر واقعی.

---

## داده

فایل `AllData.npy` همان پیکره‌ی تلگرام فارسی مرجع [۴۷]
([doi:10.17632/372RNWF9PC.1](https://doi.org/10.17632/372RNWF9PC.1)) است که از
پیش پیش‌پردازش و توکنایز شده — دقیقاً همان فایلی که در مخزن PLOT هست و مستقیم
در پایتون خوانده می‌شود (نه پایگاه‌داده لازم است نه توکنایزر). یک آرایه‌ی
۷-بخشی است که بخش *p* آن، به‌ازای هر پنجره‌ی زمانی یک لیست دارد:

| اندیس | محتوا |
|------:|-------|
| ۰ | شناسه‌ی پست‌ها |
| ۱ | پست‌های توکن‌شده |
| ۲ | نوع توکن‌ها |
| ۳ | برچسب حذف |
| ۴ | تاریخ/زمان |
| ۵ | شناسه‌ی کانال |
| ۶ | شماره‌ی پنجره (یک عدد به‌ازای هر پنجره) |

۱۰٬۲۸۴ پست، از ۱ تا ۳۱ ژانویه‌ی ۲۰۱۷. فایل در پنجره‌های **۱ ساعته** باندل شده،
اما استاندارد طلایی پنجره‌های **۱۲ ساعته** را حاشیه‌نویسی کرده است؛ بنابراین
`sskg/data.py` جریان را به ۶۲ پنجره‌ی هم‌تراز با استاندارد طلایی بازباندل
می‌کند تا شماره‌ی پنجره‌ها دقیقاً یکی شوند:

```
win = floor((t − نیمه‌شبِ روز اول) / window_hours) + 1
```

---

## پارامترها

از طریق خط فرمان (`python run_sskg.py --help`) یا `sskg/config.py`.

| پارامتر | پیش‌فرض | مقاله | معنی |
|---------|---------|-------|------|
| `--thresh-ps` | `0.5` | §۳٫۱٫۲ | `Thresh_PS` الگوریتم ۱: پست به هر موضوعی با امتیاز حداقل این مقدار می‌پیوندد. |
| `--subject-score-mode` | `softmax` | — | نحوه‌ی تبدیل خروجی دسته‌بند به `L_score`؛ یادداشت پایین را ببینید. |
| `--calibrate-thresh-ps` | خاموش | — | آستانه را طوری تنظیم می‌کند که ۷۹۵ پست بی‌برچسب بمانند (مطابق شکل ۸). |
| `--t-n` | `0.15` | §۳٫۲٫۳ | آستانه‌ی فاصله‌ی کسینوسی برای ادغام عبارت **گره** (الگوریتم ۳). |
| `--t-e` | `0.20` | §۳٫۲٫۳ | آستانه‌ی فاصله‌ی کسینوسی برای ادغام عبارت **یال** (الگوریتم ۳). |
| `--ts-seconds` | `60` | §۳٫۳ | همان `TS`، طول اسلات زمانی که بردارهای NAP و NAF روی آن ساخته می‌شوند. |
| `--t-d` | `1.2` | جدول ۳ | آستانه‌ی فراموشی روی امتیاز موجودیت. |
| `--thresh-ts` | `2.0` | جدول ۳ | `Thresh_ts` الگوریتم ۴: فاصله‌ی **اقلیدسی** بین تعبیه‌های (نرمال‌نشده‌ی) عنوان که زیر آن دو برچسب یک رخداد می‌شوند. |
| `--clustering` | `louvain` | جدول ۲ | یکی از `louvain`، `spectral`، `hcluster`. |
| `--window-hours` | `12` | §۴٫۱ | گام تشخیص؛ ۱۲ پنجره‌ها را با استاندارد طلایی هم‌تراز نگه می‌دارد. |
| `--forget-every` | `1` | شکل ۱ | تحلیل و فراموشی هر *N* پنجره اجرا شود («TimeStep»). |
| `--min-event-size` | `2` | — | خوشه‌های کوچک‌تر از این گزارش نمی‌شوند. |

وزن‌ها طبق شکل ۶ به‌صورت خطی از **۰٫۱** (قدیمی‌ترین بازه) تا **۱٫۰** (تازه‌ترین)
افزایش می‌یابند.

چند تصمیمِ تفسیری که لازم است صریح گفته شوند، چون مقاله در آن‌ها جای برداشت
باقی گذاشته است:

* **`nafr` در رابطه‌ی ۷، معکوسِ زمانیِ بردار فراوانی است**، نه معکوس عددی
  عنصربه‌عنصر. فقط با تفسیر «معکوس زمانی» هر سه سطر جدول ۱ هم‌زمان درست
  درمی‌آید — به‌ویژه یک تاریخچه‌ی کاملاً یکنواخت دقیقاً `Score = 1` می‌دهد، همان
  «در آستانه‌ی انفجار … فراوانی ذاتاً یکنواخت» مقاله. ضمناً معکوس عددی روی
  بازه‌های خالی (که فراوان‌اند) تعریف‌نشده است. در `sskg/scoring.py` هر دو
  پیاده‌سازی شده (`nafr_mode`) و حالت معکوس زمانی پیش‌فرض است.
* **`Thresh_ts = 2` و `t_d = 1.2`.** کپشن جدول ۳ و متن §۴٫۳ این دو مقدار را
  جابه‌جا گفته‌اند؛ بهترین خانه‌ی جدول ۳ (۰٫۶۷۸۶) روی `Thresh_ts = 2` و
  `t_d = 1.2` است، پس همین پیش‌فرض گرفته شده.
* **الگوریتم ۴ از فاصله‌ی اقلیدسی روی بردارهای نرمال‌نشده استفاده می‌کند.** هر
  رخداد زیرجریان با تعبیه‌ی عنوانش نمایش داده می‌شود: میانگین ساده‌ی تعبیه‌های
  خام ParsBERT سه عضو تریپل عنوان، بدون نرمال‌سازی L2 روی اعضا یا روی میانگین.
  `Thresh_ts` با فاصله‌ی اقلیدسی بین همین بردارها مقایسه می‌شود. این با فاصله‌ی
  کسینوسی الگوریتم ۳ (`t_n` و `t_e`) فرق دارد؛ آن یکی همچنان کسینوسی روی
  بردارهای نرمال‌شده است. نُرم بردارهای خام ParsBERT (mean-pooling) حدود ۱۷ تا
  ۲۰ است، پس مقیاس فاصله‌ی اقلیدسی خیلی بزرگ‌تر از فاصله‌ی کسینوسی است؛ با
  `Thresh_ts = 2` در عمل فقط عنوان‌های (تقریباً) یکسان ادغام می‌شوند.
* **تحلیل پیش از تشخیص اجرا می‌شود** (شکل ۷): امتیازدهی ← فراموشی ← خوشه‌بندی؛
  پس هر رخداد همیشه خوشه‌ای از موجودیت‌های منفجرشده است.
* **`Thresh_PS = 0.5` و برچسب چندگانه.** هدِ Persian-News با softmax آموزش دیده،
  پس حداکثر یکی از هشت گروه می‌تواند امتیاز بالای ۰٫۵ بگیرد و هر پست دست‌بالا در
  یک جریان موضوعی می‌افتد. شکل ۸ اشتراک‌های کوچکی بین مجموعه‌های موضوعی گزارش
  می‌کند که برای بازتولیدش باید لاجیت‌ها را احتمال‌های مستقل هر کلاس خواند:
  `--subject-score-mode sigmoid`. هر دو خوانش پیاده‌سازی شده و `softmax`
  پیش‌فرض است، چون چک‌پوینت این‌طور آموزش دیده است.
* **پارامترهای نمونه‌برداری مدل زبانی.** هیچ مقداری فرستاده نمی‌شود مگر خودتان
  بخواهید، پس مقادیر پیش‌فرض خودِ API اعمال می‌شوند (با `--kg-temperature`
  می‌توانید بازنویسی کنید). مدل پیش‌فرض `gpt-4o-mini` است و با `--kg-model`
  قابل تغییر است.

---

## مدل اجرا و کش‌ها

جریان پنجره‌به‌پنجره بازپخش می‌شود و برای هر پنجره سه فاز شکل ۱ (تغذیه، تحلیل،
تشخیص) اجرا می‌شود. هر چیز گران‌قیمتی کش می‌شود، پس اجرای دوم با همان پیکربندی
فقط هزینه‌ی کار روی گراف را دارد:

| کش | مسیر | چه زمانی دوباره ساخته می‌شود |
|----|------|------------------------------|
| امتیاز موضوعی | `cache/subject_scores_*.npy` | با تغییر بک‌اند یا مجموعه‌ی پست‌ها |
| گراف‌های دانش | `KnowledgeGraphs/knowledge_graphs.jsonl` | هرگز — فقط الحاقی، کلید = شناسه‌ی پست |
| امبدینگ عبارت‌ها (خام، نرمال‌نشده) | `cache/embeddings_*.npy` | با تغییر مدل امبدینگ یا قالب کش |

بهینه‌سازی‌های دیگر (چون الگوریتم‌های چاپ‌شده عمداً ساده نوشته شده‌اند):

* الگوریتم ۳ در هر UpSert کل لیست موجودیت‌ها را دوباره امبد می‌کند. اینجا هر
  عبارت **یک‌بار** امبد می‌شود، یک مسیر سریعِ تطابق دقیقِ رشته، جست‌وجو را برای
  عبارت‌های تکراری کاملاً حذف می‌کند، و نزدیک‌ترین همسایه با یک ضرب
  ماتریس-بردار BLAS روی ماتریسِ از پیش تخصیص‌یافته و نرمال‌شده به دست می‌آید؛
  ادغام هم میانگین متحرک نگه می‌دارد تا هرگز به اعضای قبلی برنگردیم.
* تریپل‌های هر پست در یک batch امبد می‌شوند، نه یک فراخوانی به‌ازای هر عبارت.
* استخراج گراف دانش در thread pool اجرا می‌شود (`--kg-workers`) با
  retryِ نمایی؛ ذخیره‌سازی به‌ازای هر پست flush می‌شود تا اجرای قطع‌شده دقیقاً
  از همان‌جا ادامه یابد.
* امتیازدهی با NumPy روی زمان‌های رخداد هر موجودیت برداری‌سازی شده است.

اگر کدِ تولیدکننده‌ی کش را عوض کردید `cache/` را پاک کنید؛ `KnowledgeGraphs/`
را فقط وقتی پاک کنید که حاضرید دوباره هزینه‌ی مدل زبانی را بدهید.

---

## خروجی

در `SystemResults/` نوشته می‌شود و پیکربندی در نام فایل مهر می‌خورد:

| فایل | محتوا | ورودیِ معیار |
|------|-------|--------------|
| `Topic_Systemresult_<stamp>.xls` | هر سطر یک رخداد: شماره‌ی پنجره + عنوان‌ها با جداکننده‌ی `\|` | **Topic Recall** |
| `ResultsToCompaire_<stamp>.xls` | یک شیت به‌ازای هر پنجره (`Window-<n>`): شناسه‌ی پست + شماره‌ی رخداد‌ها، `0` یعنی بدون رخداد | **Entropy** |
| `FinalEvents_<stamp>.tsv` | فهرست خوانا از رخدادهای سراسری | مطالعه |

هر رخداد هم عنوانِ مقاله را گزارش می‌کند (تریپلی از گراف جریان که در فضای
برداری به برچسب خوشه نزدیک‌ترین است، §۳٫۴٫۱) و هم صورت‌های سطحیِ موجودیت‌های
اصلی‌اش — چون ارزیابی عنوان‌ها را به‌صورت زیررشته در متن پست جست‌وجو می‌کند و
همین عبارت‌های تکیِ موجودیت هستند که عیناً در پست‌ها ظاهر می‌شوند.

---

## ارزیابی

```bash
cd evaluation
python evaluate.py
```

از `../SystemResults/` می‌خواند، با
`GoldenStandard/GoldenStandard_TopicID_and_TopicString.xlsx` مقایسه می‌کند و
`evaluation/Final_Evaluation_Report.xlsx` را می‌نویسد، شامل **Topic Recall**
(رابطه‌ی ۱۳: `TP / (TP + FN)`؛ یک رخداد طلایی وقتی تشخیص‌داده‌شده شمرده می‌شود
که یکی از کلماتش در عنوان‌های سیستم بیاید) و **آنتروپی کلاس / خوشه / کل**
(روابط ۹ تا ۱۲).

فایل‌های `evaluate.py` و `EvaluateFunctional.py` بدون تغییر همان اسکریپت‌های
انتشار PLOT هستند تا اعداد دو مقاله مستقیماً قابل مقایسه باشند.

نتایج مقاله روی همین پیکره (جدول ۵): Topic Recall برابر **۰٫۶۷۸۶**، آنتروپی
کلاس **۰٫۳۵۴۴**، آنتروپی خوشه ۱٫۶۸۴۶، آنتروپی کل ۱٫۰۱۹۵.

---

## تست‌ها

```bash
python -m pytest tests/ -q          # ۴۴ تست، حدود ۱ ثانیه، بدون نیاز به اینترنت
```

فایل `tests/test_units.py` روابط و ساختارهای داده را پوشش می‌دهد: این‌که تاریخچه‌ی
یکنواخت دقیقاً امتیاز ۱ و انفجار امتیاز بالاتر از ۱ می‌گیرد (جدول ۱)، این‌که NAP
تا لحظه‌ی کنونی صفرپَد می‌شود (شکل ۵)، این‌که ادغام و فراموشی گراف را سازگار نگه
می‌دارند، این‌که الگوریتم ۱ صفر/یک/چند موضوع را می‌پذیرد، و این‌که پارسرِ پاسخ
مدل زبانی از پس فرمت‌های رایج برمی‌آید.

فایل `tests/test_pipeline.py` یک `AllData.npy` مصنوعی می‌سازد که در آن یک خبرِ
کاشته‌شده دو بار ذکر و بعد منفجر می‌شود، کل خط لوله را روی آن اجرا می‌کند و
بررسی می‌کند که آن خبر بازیابی شده، ذخیره‌گاه در اجرای دوم دوباره ساخته نشده و
هر دو فایل خروجی دقیقاً شکلی را دارند که `evaluate.py` انتظار دارد.

---

## ارجاع

اگر از این کد استفاده کردید، لطفاً به مقاله‌ی SSKG (غلامی‌دستگردی،
فیضی‌درخشی، صالح‌پور — *Ain Shams Engineering Journal*، ۲۰۲۴) ارجاع دهید.
