# PaperTrail

**A Paper-to-Code Reproducibility Agent.** Point it at an ML research paper (PDF) and the GitHub
repository that implements it, then ask how a concept from the paper shows up in the code. PaperTrail
retrieves the relevant passage from the **paper** and the matching function from the **code**, reconciles
them, and returns a **verdict** — with citations to both the paper (section/page) and the code
(file/symbol/line).

The headline feature is the reconciliation verdict: PaperTrail judges the *relationship* between paper and
code, including honestly flagging what the code does **not** implement. It works both ways — paper→code and
code→paper.

## Why it's interesting

It operates over a *mixed corpus*: dense academic prose **and** source code. The two need different handling
(different embedding models, different chunking, separate vector collections), which is exactly what makes the
retrieval genuinely hard — and the point of the project.

## The verdict

Every answer is a validated structured result:

| Verdict            | Meaning                                                        |
| ------------------ | ------------------------------------------------------------- |
| `MATCH`            | The code faithfully implements the paper concept.             |
| `MATCH_SIMPLIFIED` | Implemented, but with intentional simplifications.            |
| `EXTRA_IN_CODE`    | The code does something the paper doesn't describe.           |
| `NOT_IMPLEMENTED`  | The paper concept is absent from the code.                    |
| `UNCERTAIN`        | Evidence is insufficient to decide (the agent won't guess).   |

…plus a confidence score, an explanation, and side-by-side citations to paper and code.

## Stack

| Concern            | Choice                                                              |
| ------------------ | ------------------------------------------------------------------ |
| Language / env     | Python 3.11+, managed with `uv`                                    |
| LLM                | Claude via Amazon Bedrock (temperature 0)                          |
| Paper embeddings   | Voyage `voyage-context-3` (contextualized chunk embeddings)        |
| Code embeddings    | Voyage `voyage-code-3` (code-specialized)                          |
| Vector store       | Chroma, two collections (paper / code)                             |
| Retrieval          | Hybrid: vector + BM25, fused with Reciprocal Rank Fusion           |
| Orchestration      | LangChain                                                          |
| Structured output  | Pydantic                                                           |
| Observability      | LangSmith                                                          |
| PDF parsing        | Docling                                                            |
| Code parsing       | git clone + tree-sitter (chunk by function/class)                  |
| Frontend           | Streamlit                                                          |

## Setup

```bash
# 1. Install dependencies (creates .venv)
uv sync

# 2. Configure secrets
cp .env.example .env      # then paste your AWS Bedrock, Voyage, and LangSmith keys into .env

# 3. Verify everything is wired up
uv run p2c check
```

You need: Amazon Bedrock access (a Bedrock API key **or** AWS credentials) with a Claude model enabled, a Voyage AI
API key, and a LangSmith API key. See `.env.example` for the exact variables.

## Usage

For the full architecture and a step-by-step demo walkthrough, see **[docs/GUIDE.md](docs/GUIDE.md)**.

```bash
# Verify credentials are wired up
uv run p2c check

# Ingest a paper (PDF) and the repo that implements it
uv run p2c ingest-paper path/to/paper.pdf --id attention
uv run p2c ingest-code https://github.com/karpathy/nanoGPT --id nanoGPT

# Ask how a concept maps between the two (direction auto-detected, or force it)
uv run p2c ask "How is multi-head attention from Section 3 implemented in the code?"
uv run p2c ask "Where is the Transformer encoder implemented?" --direction paper-to-code

# Web UI
uv run streamlit run app/streamlit_app.py

# Score against the hand-labeled answer key (downloads + ingests the paper & repo)
uv run python scripts/evaluate.py --ingest
```

## Reference test case

Paper *"Attention Is All You Need"* + code [nanoGPT](https://github.com/karpathy/nanoGPT). The deliberate
mismatch — nanoGPT is decoder-only while the paper is encoder-decoder — is the showcase for gap detection:
asking "where is the encoder implemented?" should return `NOT_IMPLEMENTED`.

## Results

Working end-to-end. On the reference test (*Attention Is All You Need* + nanoGPT) it scores **3/3**:

| Question | Verdict | Citation |
| --- | --- | --- |
| Multi-head attention (§3) | `MATCH_SIMPLIFIED` | `model.py` · `CausalSelfAttention` |
| Position-wise feed-forward | `MATCH_SIMPLIFIED` | `model.py` · `MLP` |
| Where is the encoder? | `NOT_IMPLEMENTED` | — (nanoGPT is decoder-only) |

The encoder case is the gap-detection showcase: the agent recognizes the paper's *bidirectional* encoder is
absent from the causal/decoder-only repo rather than forcing a false match.

**Generalization (unseen pair):** on *An Image is Worth 16×16 Words* + `google-research/vision_transformer` it also
scores **3/3** — patch embedding and the `[class]` token → `VisionTransformer.__call__`, multi-head self-attention →
`Encoder1DBlock.__call__` (~0.97–0.98 confidence) — confirming it isn't overfit to the reference case.

## Limitations & next steps

- OCR is disabled (digital-text PDFs only). On long PDFs, Docling's layout preprocessing can hit an
  out-of-memory error on later pages on low-RAM machines; those pages are skipped (early pages still ingest fine).
- One paper+repo pair per store — retrieval isn't yet scoped by paper/repo id, so use a separate `DATA_DIR` per
  pair (metadata-scoped retrieval is a planned enhancement).
- Reference following is single-hop (no full reference graph).
- Voyage's free tier without a payment method is rate-limited (~3 req/min); add a payment method (free tokens
  still apply) for smooth ingestion.

See `learnings.md` for the design decisions behind the build.
