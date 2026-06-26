# Evaluation results

PaperTrail's verdicts scored against the hand-labeled answer keys in `eval/`, using
`scripts/evaluate.py`. Each case checks the verdict label and, where relevant, that the cited code
symbol matches the expected one. Runs are at **temperature 0**, so the verdicts are reproducible.

The reference key is the *final exam* — the system was **not** tuned against it — and the second
(Vision Transformer) pair was **never seen during development**, so it tests generalization, not memorization.

---

## Reference case — "Attention Is All You Need" + nanoGPT

The deliberate mismatch: nanoGPT is **decoder-only**, while the paper is **encoder-decoder**. The
encoder question is the gap-detection showcase — the correct answer is "not implemented," not a forced match.

| # | Question | Direction | Expected | Verdict | Conf | Code symbol | Result |
| - | -------- | --------- | -------- | ------- | ---- | ----------- | ------ |
| 1 | How is multi-head attention from Section 3 implemented in the code? | paper→code | MATCH / MATCH_SIMPLIFIED | `MATCH_SIMPLIFIED` | 0.95 | `CausalSelfAttention` (`model.py`) | ✅ PASS |
| 2 | How is the position-wise feed-forward network implemented in the code? | paper→code | MATCH / MATCH_SIMPLIFIED | `MATCH_SIMPLIFIED` | 0.94 | `MLP` (`model.py`) | ✅ PASS |
| 3 | Where is the Transformer encoder implemented in the code? | paper→code | NOT_IMPLEMENTED | `NOT_IMPLEMENTED` | 0.92 | — (decoder-only) | ✅ PASS |

**Score: 3/3.** Case 3 is the headline result: the agent recognizes the paper's *bidirectional* encoder
is absent from the *causal* decoder-only repo and reports the gap honestly instead of citing a
look-alike block.

---

## Generalization — "An Image Is Worth 16×16 Words" (ViT) + google-research/vision_transformer

An **unseen** paper + repo, in a different domain (vision) and a different codebase style (JAX/Flax) —
ingested into an isolated store and scored with no further tuning.

| # | Question | Direction | Expected | Verdict | Conf | Code symbol | Result |
| - | -------- | --------- | -------- | ------- | ---- | ----------- | ------ |
| 1 | How is patch embedding implemented in the code? | paper→code | MATCH / MATCH_SIMPLIFIED | `MATCH` | 0.97 | `VisionTransformer.__call__` | ✅ PASS |
| 2 | How is the `[class]` token implemented in the code? | paper→code | MATCH / MATCH_SIMPLIFIED | `MATCH` | 0.97 | `VisionTransformer.__call__` | ✅ PASS |
| 3 | How is multi-head self-attention implemented in the code? | paper→code | MATCH / MATCH_SIMPLIFIED | `MATCH` | 0.98 | `Encoder1DBlock.__call__` | ✅ PASS |

**Score: 3/3.** The agent finds the right symbols in a codebase it never saw during development,
confirming it is not overfit to the reference case.

---

**Overall: 6/6 across both pairs.** Reproduce with:

```bash
uv run python scripts/evaluate.py --ingest --out eval/results.md \
  --title "Reference case — Attention Is All You Need + nanoGPT"
```
