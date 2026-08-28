# SinGatedLM

> **Attention-conditioned sinusoidal gating for small language models.**

SinGatedLM is an experimental language-model architecture built around a simple idea:

Instead of using attention only as an additive/residual transformation, use the attention output as a **multiplicative sinusoidal control signal** for another learned transformation.

The core operation is:

```math
f(x, y) = (Wx + b) \odot (\alpha \sin(y))
```

where:

* `x` is the current representation
* `Wx + b` is a learned linear transformation
* `y` is an attention-derived representation
* `sin(y)` provides nonlinear, oscillatory modulation
* `α` controls the gating amplitude and can be learned

---

# Architecture

The experimental language model uses the following structure:

```text
                         Token IDs
                            │
                            ▼
                     Token Embedding
                            │
                            +
                     Positional Embedding
                            │
                            ▼
                         Linear
                            │
                            ▼
                  Multi-Head Attention  ←── with residual
                            │
                            ▼
                         Linear
                            │
                            ▼
                 SinGated Attention     ←── no residual (gate must act)
                            │
                            ▼
                         Linear
                            │
                            ▼
                    Vocabulary Head
                            │
                            ▼
                          Logits
```

The central mechanism is:

```math
A = Attention(x)
```

```math
g = \alpha \sin(A)
```

```math
h = Wx + b
```

```math
SinGated(x,A) = h \odot g
```

In other words, attention is used as a **dynamic modulation signal** rather than simply being added to the representation.

## Implementation

The core operation is implemented as:

```python
class SinGatedLinear(nn.Module):
    """f(x, y) = (W @ x) * alpha * sin(y)"""

    def forward(self, x, y):
        Wx = self.linear(x)
        gate = self.alpha * torch.sin(y)
        return Wx * gate
```

The attention-conditioned block is:

```python
class SinGatedAttention(nn.Module):
    """
    A = Attention(x)
    out = SinGatedLinear(x, A)
        = (W @ x) * alpha * sin(A)
    """
```

The attention operation itself remains standard multi-head self-attention. The experimental change is what happens **after attention produces its representation**.

---

# Experiments

Experiments were performed on the **Tiny Shakespeare** character-level language-modeling task.

The purpose of these experiments is not to claim that the architecture is universally superior, but to investigate whether sinusoidal attention-conditioned modulation provides useful behavior under small parameter budgets.

---

# Experiment 1 — ~64K Parameters (Parameter-Matched)

The first experiment compares SinGatedLM against a parameter-matched PlainLM baseline.

Both models contain essentially the same number of parameters and use the same overall computational structure, with the SinGated mechanism replaced by standard learned transformations in the baseline.

### Seed 42

| Model          | Parameters | Final validation loss |
| -------------- | ---------: | --------------------: |
| PlainLM        |     64,659 |                2.6931 |
| **SinGatedLM** | **64,660** |            **2.5603** |

Difference:

```math
2.6931 - 2.5603 = 0.1328
```

SinGatedLM achieved a **0.1328 lower validation loss** while using exactly **one additional parameter**.

## Convergence

The advantage appeared throughout training rather than only at the final checkpoint.

| Iteration | SinGatedLM |    PlainLM |
| --------: | ---------: | ---------: |
|       800 |     2.7418 |     2.8728 |
|      1300 |     2.6677 |     2.7664 |
|      1600 |     2.6673 |     2.7819 |
|      2000 |     2.6180 |     2.7158 |
|      2400 |     2.6016 |     2.7247 |
|      2999 | **2.5603** | **2.6931** |

This suggests that the observed difference is not limited to a single final checkpoint and may involve differences in optimization behavior.

### Training time

```text
SinGatedLM    21.7s
PlainLM       22.0s
```

The measured training times were also very similar in this run.

---

# Experiment 2 — ~1M Parameters (Parameter-Matched)

This experiment scales both models to approximately one million parameters with **identical hidden dimensions** (`d_model = 292`).

The parameter counts are effectively identical:

```text
SinGatedLM    1,027,048 parameters
PlainLM       1,027,047 parameters
Difference              1 parameter
```

The only architectural difference is that SinGatedLM replaces the PlainLM second block (`MHA + residual + Linear`) with `SinGatedAttention` (attention-driven sinusoidal gating, no residual around the gate).

## 3,000 Steps

| Model          | Parameters | Final train loss | Final validation loss |      Time |
| -------------- | ------------: | ---------------: | --------------------: | --------: |
| **SinGatedLM** | **1,027,048** |       **1.5151** |            **1.7174** | **62.8s** |
| PlainLM        |     1,027,047 |           1.6763 |                1.8616 |     60.5s |

Validation-loss difference:

```math
1.8616 - 1.7174 = 0.1442
```

Perplexity:

```text
SinGatedLM    exp(1.7174) ≈ 5.57
PlainLM       exp(1.8616) ≈ 6.43
```

## 8,000 Steps (Convergence)

Both models were trained to convergence to observe their asymptotic behavior.

| Model          | Parameters | Final train loss | Final validation loss |     Time |
| -------------- | ------------: | ---------------: | --------------------: | -------: |
| **SinGatedLM** | **1,027,048** |       **1.3863** |            **1.5902** | **164s** |
| PlainLM        |     1,027,047 |           1.5421 |                1.7592 |    **159s** |

Validation-loss difference:

```math
1.7592 - 1.5902 = 0.1690
```

Perplexity:

```text
SinGatedLM    exp(1.5902) ≈ 4.90
PlainLM       exp(1.7592) ≈ 5.81
```

### Convergence dynamics

| Iteration | SinGatedLM Val | PlainLM Val | Gap (PL − SG) |
| --------: | -------------: | ----------: | ------------: |
|       500 |         2.3239 |      2.3334 |       −0.0095 |
|     1,000 |         2.0104 |      2.1320 |       +0.1216 |
|     2,000 |         1.8057 |      1.9725 |       +0.1668 |
|     3,000 |         1.7210 |      1.9033 |       +0.1823 |
|     4,000 |         1.6853 |      1.8664 |       +0.1811 |
|     5,000 |         1.6380 |      1.8224 |       +0.1844 |
|     6,000 |         1.6167 |      1.8034 |       +0.1867 |
|     7,000 |         1.5955 |      1.7670 |       +0.1715 |
|     8,000 |     **1.5902** |  **1.7592** |   **+0.1690** |

**Observations:**

* SinGatedLM starts slightly behind (iterations 0–500) — the sinusoidal gate creates a harder initial optimization landscape.
* It overtakes PlainLM around iteration 500 and the gap widens monotonically through ~3,000 steps.
* From iteration 3,000 to 8,000, both models improve slowly, but PlainLM appears to hit a representational floor around **1.75–1.76** while SinGatedLM continues to **1.59**.
* The last 10 validation evaluations for each model show low variance (std ≈ 0.005–0.006), indicating both have reached stable asymptotes rather than transient states.

### Generation samples at 8,000 steps

**SinGatedLM:**

```text
  Be prisons are love of the for our way,
    The subtlend shallows, for half unto an enemy.
    I as a little fight in my dutinner.
  OsLIVIA. No dear a mountakes of my fear therer horse.
  Fear. What live you you are my life, but when his prince.
  FOR. Thou love. I have me we are man. Is it once
```

**PlainLM:**

```text
        Enter CAIUSTE SERVANT. Exeunt Hame, not ekent,
    By stone, for hall when and to the tell as of any friendst,
    And to guilmes that when for a kill.
                  We were will get begles; with hall's in cunnot,
    They mince. With them and littles!
```

SinGatedLM produces more recognizable character names, dialogue structure, and grammatical fragments at the same parameter count.

---

# Experiment 3 — ~1M vs ~1.5M Parameters (Unequal)

An earlier experiment compared a ~1M SinGatedLM against a larger ~1.5M PlainLM. This is **not** parameter-matched and is included only for completeness.

| Model          |    Parameters | Final train loss | Final validation loss |      Time |
| -------------- | ------------: | ---------------: | --------------------: | --------: |
| **SinGatedLM** | **1,027,048** |       **1.6663** |            **1.8818** | **58.3s** |
| PlainLM        |     1,535,483 |           1.7800 |                1.9914 |     68.3s |

SinGatedLM used approximately **33% fewer parameters** while achieving a lower validation loss. This result should be interpreted as efficiency, not as proof of superiority at equal scale.

---

# Results Summary

```text
                         Validation Loss

~64K parameters (matched)

PlainLM                  2.6931
SinGatedLM               2.5603
                         ↓ 0.1328

~1M parameters (matched, 3,000 steps)

PlainLM                  1.8616
SinGatedLM               1.7174
                         ↓ 0.1442

~1M parameters (matched, 8,000 steps, convergence)

PlainLM                  1.7592
SinGatedLM               1.5902
                         ↓ 0.1690
                         Perplexity: 5.81 → 4.90 (1.18× better)
```

The parameter-matched experiments are the most informative:

* At **~64K**, SinGatedLM wins by **0.13** with **+1 parameter**.
* At **~1M**, SinGatedLM wins by **0.17** with **+1 parameter** after convergence.
* The advantage **scales with model size** rather than diminishing.

---

# Why SinGated?

The motivation is to explore neural transformations involving multiple interacting variables rather than repeatedly applying only:

```math
y = Wx + b
```

SinGatedLM instead uses:

```math
y = f(x, A)
```

where `A` is produced by attention.

Specifically:

```math
A = Attention(x)
```

and:

```math
f(x,A) = (Wx+b) \odot \alpha\sin(A)
```

This gives attention a second role:

```text
Traditional use:

x → Attention(x) → representation
                         │
                         ▼
                       output


SinGated use:

x ────────────────→ Linear(x) ──────┐
                                    │
Attention(x) → sin(·) → gate ──────┤
                                    ▼
                                  ×
                                    │
                                    ▼
                                  output
```

The hypothesis is that this **multiplicative nonlinear interaction** can provide useful representational behavior that is different from simply adding attention outputs to the network's representation. The sinusoid is not merely a gate — its periodicity creates multiple active regions, effectively giving the model a soft routing mechanism without explicit mixture-of-experts parameters.

---

# Experimental Philosophy

The project follows:

```text
Hypothesis
    ↓
Implementation
    ↓
Controlled experiment
    ↓
Measurement
    ↓
Scaling
    ↓
Validation
```

The goal is to test the mechanism experimentally rather than assume that a mathematically unusual operation is automatically better.

## Completed experiments

* ✅ Parameter-matched ~64K models
* ✅ Parameter-matched ~1M models (3,000 steps)
* ✅ Parameter-matched ~1M models (8,000 steps, convergence)
* ✅ Training-efficiency comparison (similar wall-clock time)

## Future experiments

* multiple random seeds
* larger datasets (WikiText-2, etc.)
* different sequence lengths
* different values of `α` (fixed vs learnable)
* alternative gating functions (`tanh`, `σ`, identity)
* removing the sinusoidal operation
* deep stack scaling (2–3 repeated blocks)
* inference-efficiency comparisons
* standard language-model benchmarks

---

# Reproducibility

The current experiments use:

* Dataset: **Tiny Shakespeare**
* Task: character-level language modeling
* Optimizer: `AdamW`
* Training iterations: `8000` (convergence runs)
* Batch size: `32`
* Block size: `128`
* Attention heads: `4`
* Learning rate: `3e-4`
* Evaluation interval: `100`
* Loss: cross-entropy
* Device: CUDA when available

The experiments use a fixed random seed for reproducibility.

Note that the exact training results can vary depending on hardware, PyTorch/CUDA versions, random-number-generator state, and other implementation details.

---

# Status

🚧 **Experimental / Research**

SinGatedLM is an experimental architecture.

The current results are encouraging:

* A parameter-matched **~64K** experiment shows a **0.1328 lower validation loss** for SinGatedLM.
* A parameter-matched **~1M** experiment shows a **0.1690 lower validation loss** for SinGatedLM after **8,000 steps** of convergence, corresponding to **1.18× lower perplexity** at effectively identical parameter count.
* The advantage **widens with scale** and **stabilizes at convergence**, suggesting a structural representational benefit rather than a transient optimization effect.

However, these results are limited to a single dataset (Tiny Shakespeare), a single random seed, and a shallow architecture. They are not sufficient to establish general superiority over Transformer architectures.

The most important next steps are **multiple-seed validation**, **alternative gating function ablations** (`sin` vs `tanh` vs `σ`), and **broader-dataset evaluation**.

---

# License

MIT License

Copyright (c) 2026 Bekhruz Suleyman

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

# Author

**Bekhruz Suleyman**
