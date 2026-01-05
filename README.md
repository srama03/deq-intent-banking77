# DEQ Intent Classification (Banking77): Baseline vs Implicit Transformer

This project studies **intent classification** on the **Banking77** dataset using:
- a **small Transformer encoder baseline** (explicit depth), and
- a **Deep Equilibrium Model (DEQ)** formulation using an **implicit Transformer block** (equilibrium / fixed-point layer).

The goal is not to chase SOTA. The goal is to compare **explicit vs implicit depth** in a realistic NLP task, and to analyze robustness and failure modes.

---

## Task & dataset

### What problem
Given a short user utterance (single sentence), predict one of **77 intent classes** (e.g., card issues, payments, transfers). This mirrors real-world NLP routing systems used in assistants and customer support.

### Why this dataset
Banking77 provides short, natural-language queries across many intent classes, making it a clean benchmark for architectural comparison without multi-turn dialogue complexity.

---

## Models

### Baseline (explicit depth)
Small Transformer encoder trained from scratch:
- subword tokenization (WordPiece/BPE)
- token + positional embeddings
- **3-layer Transformer encoder** (Pre-LN)
- **mean pooling** over token states
- linear classifier over 77 intents

**Config (locked)**
- max_len: 64  
- d_model: 256  
- n_heads: 4  
- d_ff: 1024  
- dropout: 0.1  
- activation: GELU  
- layers: 3  
- pooling: mean  

### DEQ (implicit depth)
Replace the stacked encoder with a single implicit Transformer block and solve for an equilibrium representation:

\[
z^\* = f_\theta(z^\*, x)
\]

Inference uses fixed-point iteration until convergence or a max-iteration cap. Training uses implicit differentiation (no backprop through every iteration explicitly).

**Solver policy (locked)**
- Solver: fixed-point iteration (Picard), init \( z_0 = x \)
- Stop when relative residual < 1e-3 or max iters reached
- max_iters: 25 (train), 50 (inference)
- log convergence stats and residuals

---

## Evaluation

### Metrics
- Accuracy
- Macro-F1

### Robustness: noisy/OOV stress test (locked)
Evaluate on:
1) clean test set  
2) noisy test set generated with light perturbations (typos, stopword drop, texting noise)

Report clean vs noisy metrics + relative drop.

### Error taxonomy (locked)
Analyze misclassifications using:
- top confusion pairs
- representative error examples
- supergroup-level confusions (9 manual bins)

---

## Explicit non-goals
- No LLM fine-tuning
- No multi-turn dialogue
- No slot filling / NER
- No exhaustive hyperparameter sweeps
- No SOTA claims

---

## Repo layout (planned)

- `src/` — training, eval, analysis scripts  
- `configs/` — baseline/deq configs  
- `notebooks/` — plots + inspection only  
- `results/` — local run artifacts (not tracked by git)

---

## Planned outputs
- clean vs noisy comparison table (baseline vs DEQ)
- DEQ convergence statistics
- confusion matrix (top intents)
- top confusion pairs + qualitative examples

