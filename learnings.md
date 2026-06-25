# Learnings

Each non-trivial decision recorded as **Decision → Why → Alternative considered**. Newest at the bottom.

---

### Project is its own git repository (not a subfolder of the catch-all parent repo)
- **Decision:** `git init` this folder as a standalone repo and push to a dedicated public GitHub repo.
- **Why:** Clean, portfolio-ready history; the `.gitignore`-first commit guarantees no secret ever enters history;
  a focused repo is what a recruiter actually looks at.
- **Alternative:** Commit into the existing `ali-boy-ships` catch-all repo — rejected as messy and mixing
  unrelated projects.

### `.gitignore` (with `.env`) is the first commit
- **Decision:** Commit `.gitignore` alone as the root commit before any other file exists.
- **Why:** Makes it structurally impossible for a key to enter git history; `.env` is ignored from commit #1.
- **Alternative:** Add `.gitignore` alongside the first batch of code — rejected; leaves a window where `.env`
  could be staged.

### Two embedding models, two Chroma collections, LLM as the only bridge
- **Decision:** Embed paper text with `voyage-context-3` and code with `voyage-code-3`, store in separate Chroma
  collections, and cross sides only by having the LLM reformulate the query for the target side.
- **Why:** Paper vectors and code vectors live in different spaces and aren't directly comparable; reformulation
  keeps every similarity search within a single embedding space.
- **Alternative:** A single shared embedding model / collection — rejected; loses code-specialization and produces
  meaningless cross-space comparisons.

### Precompute embeddings; don't use Chroma's `embedding_function`
- **Decision:** Call Voyage ourselves and pass vectors to Chroma via `collection.add(embeddings=...)`.
- **Why:** `voyage-context-3` embeds a chunk *in the context of its sibling chunks* via `contextualized_embed`,
  which takes nested lists — this can't be expressed through Chroma's per-text `embedding_function` signature.
- **Alternative:** Register a Chroma embedding function — rejected; incompatible with contextualized embeddings.

### Deterministic orchestrated pipeline, not a free-form ReAct agent
- **Decision:** A fixed loop — decompose → retrieve source side → reformulate → retrieve target side →
  reference-follow → reconcile → verdict — in plain LangChain at temperature 0.
- **Why:** Reproducible verdicts and an explainable flow; matches the brief's "no LangGraph for V1".
- **Alternative:** A tool-calling/ReAct agent that decides its own steps — rejected as less reproducible and
  harder to explain.

### LangSmith tracing is required (not optional)
- **Decision:** Tracing is on for every run; a LangSmith key is part of required setup.
- **Why:** The whole value of an agent demo is being able to show *why* it produced a verdict — the trace of
  retrieval + reconciliation is the evidence.
- **Alternative:** Treat tracing as optional/dev-only — rejected per project requirement.

### `uv` installed via pip (Windows)
- **Decision:** `python -m pip install -U uv`; it landed on PATH directly.
- **Why:** Simplest path on this machine where Python's Scripts dir is already on PATH; avoids the standalone
  installer's separate PATH handling.
- **Alternative:** The astral standalone installer — works too, but adds a PATH step for the current shell.

### tree-sitter node API is method-based; `parse()` takes `str`
- **Decision:** Code chunking adapts to the installed binding where node accessors are *methods*
  (`node.kind()`, `node.child(i)`, `node.start_byte()`, `node.start_position().row`) and `Parser.parse()`
  takes a `str` (byte offsets index its UTF-8 encoding). A thin adapter hides this from the walk logic.
- **Why:** The bundled binding (tree-sitter 0.25.x via `tree-sitter-language-pack`) diverges from the classic
  py-tree-sitter property API; found by introspection after a failing test, not by guessing from docs.
- **Alternative:** Pin classic py-tree-sitter + per-language grammar packages — rejected; the language pack
  ships many grammars as prebuilt (Windows-friendly) wheels, well worth one small adapter.

### Disabled Docling OCR for paper ingestion
- **Decision:** Build the `DocumentConverter` with `PdfPipelineOptions(do_ocr=False)`.
- **Why:** Research papers are digital text, so OCR adds no value — only latency and memory. Layout parsing still
  runs and produces clean chunks.
- **Caveat (honest):** disabling OCR did **not** eliminate the `std::bad_alloc` seen on this machine — it still
  occurs during *layout* preprocessing on the later pages of long PDFs (memory limit while rendering page images).
  It's non-fatal: Docling skips those pages and the early pages (which held the queried content) ingest fine
  (41 chunks for Attention, 48 for ViT). A full fix would lower the page-image scale or use a higher-memory box.
- **Alternative:** Keep default OCR — rejected; pure cost for zero benefit on digital PDFs.

### Generalization holds: 3/3 on an unseen paper+repo (ViT)
- **Decision:** Validated on *An Image is Worth 16x16 Words* + `google-research/vision_transformer` (never seen
  during development), in an isolated data store. Scored 3/3: patch embedding and `[class]` token →
  `VisionTransformer.__call__`, multi-head self-attention → `Encoder1DBlock.__call__` (0.97–0.98 confidence).
- **Why it matters:** confirms the system isn't overfit to the Attention/nanoGPT reference case, and that the
  earlier reconcile-prompt discernment rule generalizes (it found the right symbols in a JAX/Flax codebase with a
  totally different style).
- **Note:** the store doesn't yet filter retrieval by paper_id/repo_id, so each paper+repo pair uses its own data
  dir (`DATA_DIR=...`). Metadata-scoped retrieval (one store, many pairs) is a clean future enhancement.

### Reference test passes 3/3; gap detection needed a general discernment rule
- **Decision:** The first live run scored 2/3 — multi-head attention → `CausalSelfAttention` and FFN → `MLP`
  matched correctly, but "where is the encoder?" returned `MATCH_SIMPLIFIED` (citing nanoGPT's causal `Block`).
  Added one general rule to the reconcile prompt: a structurally similar component is not a match unless the
  queried concept's *distinguishing* features are present; otherwise return `NOT_IMPLEMENTED`. Re-run scored 3/3
  (encoder → `NOT_IMPLEMENTED`, confidence 0.92, explanation citing bidirectional-vs-causal).
- **Why:** Honest gap detection is the headline feature; the model was conflating a decoder block with an encoder.
- **Honesty note / Alternative:** The change came *after* seeing the answer key, which risks teaching-to-the-test.
  It's mitigated by keeping the rule fully general (no mention of "encoder"/nanoGPT) — but the real proof of
  non-overfitting is the generalization run on an unseen paper+repo. Hardcoding the encoder answer was rejected.

---

## Operating notes — tracking AWS cost & free credits

Running on the **$100 AWS sign-up promotional credit**. What actually costs money in this project is **only the
Bedrock LLM calls** — embeddings run on Voyage (separate free tier) and Docling/tree-sitter run locally. Ingesting a
paper or repo makes **no** Bedrock calls; only asking a question does (decompose → reformulate → reconcile ≈ a few
calls per question). So the credit is plenty for development and the demo.

### Step 1 — Set a budget alert FIRST (before any heavy use)
1. Open **AWS Budgets**: https://console.aws.amazon.com/billing/home#/budgets
2. **Create budget** → **Cost budget** → period **Monthly**.
3. Set an amount (e.g. **$15**). Add alert thresholds at **50%, 80%, 100%** of actual cost → enter your email.
4. (Optional, strongest safety net) Also create a **Zero-spend budget** — it emails you the moment any real charge appears.

> Why this matters: the $100 is a *credit*, not a hard cap. When it runs out, normal charges begin silently. A budget
> alert is what tells you before that happens.

### Step 2 — Watch the remaining credit balance
- **Credits page**: https://console.aws.amazon.com/billing/home#/credits — shows credit **amount remaining** and the
  **expiration date** (sign-up credits usually expire after ~6–12 months, so use it before then).

### Step 3 — See what you've spent and on what
- **Bills** (current month by service): https://console.aws.amazon.com/billing/home#/bills
- **Cost Explorer** (trend + breakdown): https://console.aws.amazon.com/cost-management/home#/cost-explorer
  - First visit: click **Enable Cost Explorer** (data can lag up to ~24h).
  - Filter **Service = Amazon Bedrock**; group by **Usage Type** to see input vs output token cost.

### Reading it correctly (credits vs charges)
- Cost Explorer / Budgets show **gross charges**; your credit is applied **separately** on the bill. So you may see a
  "charge" in Cost Explorer that the credit then fully covers.
- To track **how much credit is left**, use the **Credits page** (Step 2).
- To avoid surprises **after** credits run out, rely on the **budget alert** (Step 1).

### Rough cost intuition for this project
- Bedrock bills per 1K input/output tokens. **Sonnet 4.6** is the cost-effective choice; **Opus 4.8** is roughly ~5×
  pricier for the same tokens. Temperature 0 doesn't change cost.
- A single question here is a handful of cents on Sonnet (more on Opus). Ingestion is free of Bedrock cost.
