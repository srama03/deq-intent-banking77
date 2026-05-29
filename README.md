# Intent Classification: Explicit Depth vs. Fixed-Point Equilibrium Models
*Comparing a 3-layer Transformer baseline, a DEQ-like weight-tied encoder, and a true Deep Equilibrium Model with implicit differentiation on Banking77.*

## Motivation
Deep Equilibrium Models (Bai et al., 2019) are theoretically elegant—infinite implicit depth, constant memory, and principled gradients via the Implicit Function Theorem. But theory and practice often diverge. This project was about closing that gap: implementing DEQ from scratch, understanding the core mechanisms, and honestly characterizing where the theoretical advantages show up, where they don't, and why.

## Dataset
The goal of this project is to isolate the effect of the architecture. For this purpose, Banking77, a HuggingFace dataset, was chosen. It's a clean, well-understood benchmark with enough classes (77) to make classification non-trivial, but simple enough that data preprocessing doesn't obscure the architectural comparison. A sequence classification task was chosen as a controlled testbed—sequence modelling is where DEQ's implicit depth is theoretically motivated, making it a natural domain to test whether the theoretical advantages materialize in practice.

## Models

### Baseline
A 3-layer Pre-LN Transformer encoder trained from scratch, serving as the explicit-depth baseline. Uses GELU activations, mean pooling over token states, and a linear classifier over 77 intents. Achieved 83.8% macro-F1 on clean, 74.3% on noisy.

### DEQ-like (Ablation)
As an ablation between explicit and implicit depth, a weight-tied encoder is iterated to a fixed point z\* = f(z\*, x) via damped fixed-point iteration. Same architecture as the baseline, but with a single shared layer instead of 3 distinct ones. Gradients backpropagate through the unrolled iterations. Achieved 84.1% macro-F1 on clean, 74.5% on noisy.

### True DEQ
The true DEQ uses the same fixed-point forward pass as the weight-tied model, but replaces unrolled backpropagation with implicit differentiation via the Implicit Function Theorem. A custom `torch.autograd.Function` intercepts the backward pass, solving $(I - J_f^T)v = g$ to compute parameter gradients directly at equilibrium—no solver iterations stored in the computation graph. Achieved 82.8% macro-F1 on clean, 73.4% on noisy.

## Results

*Noise: synthetic perturbations applied to the test set—random stopword deletion and random character deletion, each with probability 0.1.*


| Model    | Clean F1 | Noisy F1 | Drop |
|----------|----------|----------|------|
| Baseline | 83.8%    | 74.3%    | 9.4% |
| DEQ-like | 84.1%    | 74.5%    | 9.6% |
| True DEQ | 82.8%    | 73.4%    | 9.4% |

All three models drop ~9.5% under noise, so the architecture doesn't really affect robustness here.

The baseline makes the most sensible mistakes—when it's wrong, it's wrong on genuinely similar intents. DEQ-like has the highest F1 (84.1%) but introduces some odd confusions that the baseline doesn't make (e.g. `pin_blocked` vs `get_physical_card`), which suggests its representations aren't quite as clean despite the better number.

True DEQ has the lowest F1, but its error patterns overlap a lot with the baseline — it's not fundamentally broken, just limited by gradient quality. DEQ-like beats it because backpropping through unrolled iterations gives cleaner gradients than a partially-solved implicit backward system.

## Key Findings
- DEQ works only under the right conditions—naive implementation on a small dataset won't show its advantages
- Noise robustness was consistent across all three architectures—architectural differences didn't affect how models handle noisy input
- Raw F1 doesn't tell the full story—DEQ-like has the best F1 but makes more semantically inconsistent errors than the baseline. True DEQ has the lowest F1 but its error patterns closely resemble the baseline's, suggesting the representation quality is intact despite the gradient approximation

## Limitations
- Uses naive Picard fixed-point iteration instead of Anderson/Broyden acceleration, which is standard in DEQ implementations and would give cleaner implicit gradients
- Banking77 is too small to surface DEQ's core advantage — the memory efficiency of constant-depth representations doesn't matter on 10k examples

## Future Work
- Implement Anderson acceleration for the backward solver and test whether implicit gradients improve
- Run on a larger sequence modeling task with GPU access to see DEQ's memory efficiency in practice
- Explore whether better solver convergence closes the gap between true DEQ and DEQ-like

## Demo
[Live on HuggingFace Spaces](https://huggingface.co/spaces/srama03/deq-intent-banking77)

## Setup
```bash
# setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# train
python -m src.train --config configs/baseline.yaml
python -m src.train --config configs/deq.yaml
python -m src.train --config configs/deq-implicit.yaml

# evaluate
python -m src.evaluate
python -m src.analyze_errors
```
