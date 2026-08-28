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
                  Multi-Head Attention
                            │
                            ▼
                         Linear
                            │
                            ▼
                 SinGated Attention
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

Experiments were initially performed on the **Tiny Shakespeare** character-level language-modeling task.

The purpose of these experiments is not to claim that the architecture is universally superior, but to investigate whether sinusoidal attention-conditioned modulation provides useful behavior under small parameter budgets.

---

# Experiment 1 — ~64K Parameters

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

The parameter difference is therefore negligible:

```text
SinGatedLM    64,660
PlainLM       64,659
Difference         1
```

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

# Experiment 2 — ~1M vs ~1.5M Parameters

The architecture was then scaled to approximately one million parameters.

This experiment was **not parameter-matched**.

The SinGatedLM contained approximately 1.03M parameters, while the PlainLM contained approximately 1.54M parameters.

### Results

| Model          |    Parameters | Final train loss | Final validation loss |      Time |
| -------------- | ------------: | ---------------: | --------------------: | --------: |
| **SinGatedLM** | **1,027,048** |       **1.6663** |            **1.8818** | **58.3s** |
| PlainLM        |     1,535,483 |           1.7800 |                1.9914 |     68.3s |

Validation-loss difference:

```math
1.9914 - 1.8818 = 0.1096
```

Parameter ratio:

```math
\frac{1,027,048}{1,535,483} \approx 0.669
```

Thus, in this experiment, SinGatedLM used approximately **33% fewer parameters** while achieving a lower validation loss.

It also completed the 3000-iteration training run faster:

```text
SinGatedLM    58.3s
PlainLM       68.3s
```

### Important limitation

This is **not** an equal-parameter comparison.

The result therefore should be interpreted as:

> SinGatedLM achieved lower validation loss than the tested PlainLM configuration while using substantially fewer parameters.

It should **not** be interpreted as proof that SinGatedLM is superior to an equally-sized baseline at the ~1M scale.

A parameter-matched ~1M experiment is the next important comparison.

---

# Results So Far

The current experiments show:

```text
                         Validation Loss

~64K parameters

PlainLM                  2.6931
SinGatedLM               2.5603
                         ↓ 0.1328


~1M / ~1.5M parameters

PlainLM                  1.9914
SinGatedLM               1.8818
                         ↓ 0.1096
```

The first experiment is particularly useful because the models were effectively parameter-matched:

```text
SinGatedLM    64,660 parameters
PlainLM       64,659 parameters
```

while the ~1M experiment demonstrates that the architecture continues to produce competitive results at a substantially larger scale, although that comparison uses different parameter budgets.

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

The hypothesis is that this multiplicative nonlinear interaction can provide useful representational behavior that is different from simply adding attention outputs to the network's representation.

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

Future experiments include:

* parameter-matched ~1M models
* multiple random seeds
* larger datasets
* different sequence lengths
* different values of `α`
* learnable vs fixed `α`
* alternative gating functions
* removing the sinusoidal operation
* replacing `sin` with other nonlinear functions
* training-efficiency comparisons
* inference-efficiency comparisons
* standard language-model benchmarks

---

# Reproducibility

The current experiments use:

* Dataset: **Tiny Shakespeare**
* Task: character-level language modeling
* Optimizer: `AdamW`
* Training iterations: `3000`
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

* A parameter-matched ~64K experiment shows a **0.1328 lower validation loss** for SinGatedLM.
* A ~1M vs ~1.5M experiment shows a **0.1096 lower validation loss** for SinGatedLM despite using approximately **33% fewer parameters**.

However, these results are not sufficient to establish general superiority over Transformer architectures.

The most important next step is a **parameter-matched ~1M experiment**, followed by multiple-seed and broader-dataset evaluation.

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
