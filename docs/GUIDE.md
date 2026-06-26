# PaperTrail — How It Works & Demo Guide

A complete, in-depth walkthrough: what the system does, how every piece works, the exact data flow for a
question, and step-by-step instructions to run the demo (CLI and the Streamlit UI) — including your own paper+repo.

---

## 1. What it does

You give PaperTrail two things:

1. An **ML research paper** (PDF).
2. The **GitHub repository** that implements it.

You then ask a question like *"How is multi-head attention from Section 3 implemented in the code?"* PaperTrail:

- finds the relevant passage in the **paper**,
- finds the matching function/class in the **code**,
- **reconciles** them and returns a **verdict** with citations to both sides.

The verdict is one of:

| Verdict | Meaning |
| --- | --- |
| `MATCH` | The code faithfully implements the paper concept. |
| `MATCH_SIMPLIFIED` | Implemented, but with deliberate simplifications. |
| `EXTRA_IN_CODE` | The code does something the paper doesn't describe. |
| `NOT_IMPLEMENTED` | The paper concept is absent from the code. |
| `UNCERTAIN` | Evidence is insufficient to decide (it won't guess). |

…plus a **confidence** (0–1), an **explanation**, and **side-by-side citations** (paper section/page · code
file/symbol/lines). It works **both directions**: paper→code and code→paper.

---

## 2. Architecture (the big picture)

The core difficulty is a **mixed corpus**: dense prose *and* source code need different handling. So there are
**two independent pipelines** and **two vector collections**, and the **LLM is the only bridge** between them.

```
  PAPER (PDF)                                    CODE (git repo)
      │                                                │
  Docling parse                                  git clone (shallow)
  (OCR off, layout-aware)                        tree-sitter parse
      │                                                │
  HybridChunker                                  chunk by function / class
  (section + page aware)                         (file, symbol, line range)
      │                                                │
  voyage-context-3                               voyage-code-3
  (contextualized chunk embeddings)              (code-specialized embeddings)
      │                                                │
  Chroma "paper" collection                      Chroma "code" collection
  + BM25 index (rebuilt on load)                 + BM25 index (rebuilt on load)
      └───────────────┐                ┌───────────────┘
                      ▼                ▼
                 HYBRID RETRIEVAL (each side)
            vector + BM25 → Reciprocal Rank Fusion
                          │
                  ┌───────┴────────────────────────────────┐
                  │            THE AGENT LOOP                │
                  │  (plain LangChain, deterministic, T=0)   │
                  │                                          │
                  │  1. plan: detect direction + concept     │
                  │  2. retrieve SOURCE side (hybrid)        │
                  │  3. reference-follow (paper, 1 hop)      │
                  │  4. reformulate query for TARGET side    │
                  │  5. retrieve TARGET side (hybrid)        │
                  │  6. reconcile → structured Verdict       │
                  └───────────────────┬──────────────────────┘
                                      ▼
                       Verdict (Pydantic) + citations
                       (traced in LangSmith)
```

**Why two embedding models / two collections?** Paper vectors (from voyage-context-3) and code vectors (from
voyage-code-3) live in different vector spaces — comparing them directly is meaningless. So we never do that.
To cross from one side to the other, **the LLM rewrites the query for the target side**, and that text is embedded
with the *target* side's model and searched against the *target* collection.

---

## 3. Component-by-component

### 3.1 Paper ingestion — `src/paper_to_code/ingestion/paper.py`
- **Docling** `DocumentConverter` parses the PDF in a layout-aware way (headings, paragraphs, tables). OCR is
  **disabled** (papers are digital text; OCR i/s pure cost).
- **HybridChunker** splits the document into token-sized chunks that respect section boundaries.
- For each chunk we record metadata: `section` (heading path), `page`, and any cross-reference labels found in the
  text (`equations`, `figures`, `section_refs`). These drive both **citations** and **reference-following**.
- Output: a list of `Chunk(source="paper", …)`.

### 3.2 Code ingestion — `src/paper_to_code/ingestion/code.py`
- Shallow-clones the repo (or reads a local path), walks source files (skipping `node_modules`, `.git`, etc.).
- **tree-sitter** parses each file; we walk the syntax tree and emit one chunk per **function, class, and method**
  (classes are captured whole *and* split into methods — coarse + fine granularity).
- Metadata: `file`, `symbol` (qualified, e.g. `CausalSelfAttention.forward`), `kind`, `language`,
  `start_line`/`end_line` — exactly what a code citation needs.
- Output: a list of `Chunk(source="code", …)`.

### 3.3 Embeddings — `src/paper_to_code/embeddings/`
- **Paper:** `voyage-context-3` via `contextualized_embed` — each chunk is embedded *aware of its sibling chunks*
  in the same document, which improves retrieval without manual context-stuffing.
- **Code:** `voyage-code-3` via the standard `embed` — specialized for source code.
- Requests are **batched** under a token budget (kept small to fit Voyage's free-tier limit) with **retry +
  backoff** on rate-limit/transient errors.

### 3.4 Vector store — `src/paper_to_code/store/vector_store.py`
- **Chroma**, two collections (`paper`, `code`), cosine space.
- We **precompute embeddings** and hand the vectors to Chroma directly (no Chroma `embedding_function`) — because
  the contextualized API embeds *groups* of chunks, which the per-text function can't express.

### 3.5 Hybrid retrieval — `src/paper_to_code/retrieval/hybrid.py`
- **Vector search** (semantic) catches meaning; **BM25** (lexical, via `rank_bm25`) catches exact tokens like
  symbol names and equation labels. The tokenizer splits identifiers (`CausalSelfAttention` → `causal self
  attention`) so code names match prose.
- The two ranked lists are fused with **Reciprocal Rank Fusion** (RRF): `score = Σ 1/(k + rank)`. RRF combines by
  *rank*, sidestepping the problem that vector similarity and BM25 scores aren't on the same scale.
- BM25 is rebuilt in memory from Chroma's stored documents on load — no separate index to persist.

### 3.6 The agent loop — `src/paper_to_code/agent/loop.py`
A **deterministic, explainable pipeline** (not a free-form ReAct agent), run at **temperature 0** so the same
inputs give the same verdict:

1. **Plan** (`reformulate.plan_query`): the LLM classifies the **direction** (paper→code vs code→paper) and
   extracts the **core concept**.
2. **Retrieve the source side** with hybrid retrieval.
3. **Reference-follow** (`retrieval/reference.py`): if a top paper chunk says "see Equation 3 / Section 4.2", do
   one extra retrieval for that element and add it to the evidence (single hop).
4. **Reformulate** (`reformulate.reformulate`): the LLM rewrites the query for the **target** side (e.g. a paper
   concept → code-ish keywords like "attention head softmax qkv").
5. **Retrieve the target side** with hybrid retrieval on the reformulated query.
6. **Reconcile** (`reconcile.reconcile`): the LLM is given both sides' evidence and returns the structured
   `Verdict`. The prompt insists on grounding in the evidence, prefers `UNCERTAIN` over guessing, and tells it
   that a *structurally similar* component is not a match unless the queried concept's distinguishing features are
   present (→ honest `NOT_IMPLEMENTED`).

### 3.7 The verdict — `src/paper_to_code/models.py`
A validated Pydantic model produced via Bedrock's structured output (`with_structured_output(Verdict,
include_raw=True)`). Fields: `verdict`, `confidence`, `direction`, `explanation`, `paper_citation`
(section/page/equations/figures/quote), `code_citation` (file/symbol/lines/quote). If structured parsing ever
fails, the loop falls back to `UNCERTAIN` instead of crashing.

### 3.8 LLM + observability — `src/paper_to_code/llm.py`
- **Claude (Amazon Bedrock)**, `ChatBedrockConverse`, temperature 0. Auth via a Bedrock API key (bearer token) or
  AWS access keys; model id is `BEDROCK_MODEL_ID` (default Sonnet 4.6).
- **LangSmith** tracing is on when a valid key is present (gated by a one-time validation so an invalid key never
  spams errors). Every run shows up as a trace you can inspect.

---

## 4. What happens for one question (concrete)

Question: *"How is multi-head attention from Section 3 implemented in the code?"*

1. **Plan** → direction `paper_to_code`, concept `"multi-head attention"`.
2. **Retrieve paper** for "multi-head attention" → the §3.2 "Scaled Dot-Product / Multi-Head Attention" chunks.
3. **Reference-follow** → if those chunks cite an equation, pull it in too.
4. **Reformulate for code** → e.g. `"attention head split q k v softmax projection causal mask"`.
5. **Retrieve code** for that → `CausalSelfAttention` (and its `forward`).
6. **Reconcile** → `MATCH_SIMPLIFIED`, confidence ~0.9, paper cite §3.2 / code cite `model.py ·
   CausalSelfAttention`, with an explanation grounded in both.

---

## 5. How to run it

### One-time setup
```bash
uv sync                       # create the venv + install deps (already done)
cp .env.example .env          # then paste your keys (already done)
uv run p2c check              # should show Bedrock / Voyage / LangSmith all OK
```

### CLI
```bash
# Ingest a paper and the repo that implements it
uv run p2c ingest-paper path/to/paper.pdf --id mypaper
uv run p2c ingest-code  https://github.com/owner/repo --id myrepo

# Ask (direction auto-detected; or force it)
uv run p2c ask "How is X from Section N implemented in the code?"
uv run p2c ask "What paper concept does class Foo implement?" --direction code-to-paper

# Useful flags:  --k 8   (chunks per side)   --no-follow-refs   (skip reference following, faster)
```

### Streamlit UI (use this for the demo recording)
```bash
uv run streamlit run app/streamlit_app.py
```
Then in the browser: **sidebar →** upload the PDF, click *Ingest paper*; paste the repo URL, click *Ingest repo*;
**main panel →** type your question, pick a direction, click **Ask**. You get the verdict card with side-by-side
paper/code citations and an expander showing the raw retrieved chunks.

---

## 6. Demo workflow (step by step, for recording)

1. `uv run p2c check` — show all three services green (proves it's really wired to Bedrock/Voyage/LangSmith).
2. `uv run streamlit run app/streamlit_app.py`.
3. Upload the paper PDF → *Ingest paper* (note the chunk count).
4. Paste the repo URL → *Ingest repo* (note the chunk count).
5. Ask the "happy path" question (a concept that IS implemented) → show `MATCH`/`MATCH_SIMPLIFIED` with citations.
6. Ask the "gap" question (a concept that ISN'T implemented) → show `NOT_IMPLEMENTED` — the honesty showcase.
7. (Optional) Open LangSmith → show the trace of retrieval + reconciliation for that run.

> Tip for the recording: ingest **before** you hit record (ingestion is the slow part on the free Voyage tier),
> then record the question-and-verdict part, which is fast.

---

## 7. Running your own / a third test case

Many paper+repo pairs can live in **one** store: every chunk is tagged with its `paper_id`/`repo_id`, and
selecting an active pair scopes **both** the vector and BM25 sides so evidence never mixes. Just ingest and scope:

```bash
uv run p2c ingest-paper path/to/your_paper.pdf --id yourpaper
uv run p2c ingest-code  https://github.com/owner/yourrepo --id yourrepo

uv run p2c list                                  # everything ingested, with chunk counts
uv run p2c ask "your question here" --paper yourpaper --repo yourrepo
uv run p2c remove-paper yourpaper                # tidy up (remove-repo too; -y skips the prompt)
```

In the Streamlit UI the sidebar does the same: a **Library** panel lists papers/repos with delete buttons, and the
**Active scope** dropdowns choose the pair every question is scoped to. With nothing selected, retrieval searches
everything (backward-compatible). You can still isolate a case entirely by pointing `DATA_DIR` at a fresh folder,
but it's no longer required.

The evaluation harness can score a labelled set for any pair:
```bash
$env:DATA_DIR='data/mycase'
uv run python scripts/evaluate.py --ingest --no-follow-refs --key eval/your_key.yaml
```
(Model the YAML on `eval/answer_key.yaml`.)

---

## 8. Results so far

**Reference — Attention Is All You Need + nanoGPT (3/3):**

| Question | Verdict | Conf | Citation |
| --- | --- | --- | --- |
| Multi-head attention (§3) | `MATCH_SIMPLIFIED` | 0.88 | `model.py` · `CausalSelfAttention` |
| Position-wise feed-forward | `MATCH_SIMPLIFIED` | 0.90 | `model.py` · `MLP` |
| Where is the encoder? | `NOT_IMPLEMENTED` | 0.92 | — (decoder-only) |

**Generalization — ViT + google-research/vision_transformer (3/3, never seen during dev):**

| Question | Verdict | Conf | Citation |
| --- | --- | --- | --- |
| Patch embedding | `MATCH` | 0.98 | `VisionTransformer.__call__` |
| Multi-head self-attention | `MATCH` | 0.97 | `Encoder1DBlock.__call__` |
| `[class]` token | `MATCH` | 0.98 | `VisionTransformer.__call__` |

---

## 9. Limitations & troubleshooting

- **Voyage free tier** is ~3 requests/min (no payment method needed — your 200M free tokens still apply). Ingestion
  is just slower. A payment method lifts the rate limit at no cost within the free tokens.
- **Docling OOM on long PDFs:** layout preprocessing can fail (`std::bad_alloc`) on later pages on low-RAM
  machines. It's non-fatal — those pages are skipped and the early pages (which usually hold the queried content)
  ingest fine.
- **Many pairs per store:** retrieval is scoped by `paper_id`/`repo_id`, so one store holds many pairs without
  evidence mixing — select an active pair (CLI `--paper`/`--repo`, or the sidebar). See §7.
- **LangSmith key:** must be from the **US** region for the default endpoint (`p2c check` confirms it live).

For the *why* behind each design choice, see `learnings.md`.
