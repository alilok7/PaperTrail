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

You need: AWS credentials with a Claude (Sonnet-class) model enabled in Bedrock, a Voyage AI API key, and a
LangSmith API key. See `.env.example` for the exact variables.

## Reference test case

Paper *"Attention Is All You Need"* + code [nanoGPT](https://github.com/karpathy/nanoGPT). The deliberate
mismatch — nanoGPT is decoder-only while the paper is encoder-decoder — is the showcase for gap detection:
asking "where is the encoder implemented?" should return `NOT_IMPLEMENTED`.

## Status

🚧 In active development. See `agent-status.md` for current progress and `learnings.md` for the design
decisions behind the build.
