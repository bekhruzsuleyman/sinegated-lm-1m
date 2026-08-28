# SinGatedLM

> **Attention-conditioned sinusoidal gating for small language models.**

SinGatedLM is an experimental language-model architecture built around a simple idea:

Instead of using attention only as an additive/residual transformation, use the attention output as a **multiplicative sinusoidal control signal** for another learned transformation.

The core operation is:

$$
f(x,y) = (Wx+b)\odot\left(\alpha\sin(y)\right)
$$

where:

* \(x\) is the current representation
* \(W x+b\) is a learned linear transformation
* \(y\) is an attention-derived representation
* \(\sin(y)\) provides nonlinear, oscillatory modulation
* \(\alpha\) controls the gating amplitude and can be learned

## Architecture

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
          ┌─────────────────────┐
          │  SinGated Attention │
          │                     │
          │ A = Attention(x)    │
          │                     │
          │ g = α · sin(A)      │
          │                     │
          │ out = Linear(x) ⊙ g │
          └─────────────────────┘
                    │
                    ▼
                Linear
                    │
                    ▼
               Vocabulary
                    │
                    ▼
                  Logits
```

The central mechanism can be summarized as:

$$
A = \operatorname{Attention}(x)
$$

$$
g = \alpha\sin(A)
$$

$$
h = \operatorname{Linear}(x)
$$

$$
\boxed{\operatorname{SinGated}(x,A)=h\odot g}
$$

This allows the attention mechanism to act as a **dynamic modulation signal** rather than simply being added to the representation.

---

# Experiments

The initial experiments were performed on the **Tiny Shakespeare** character-level language-modeling task.

The goal was not to build a production-scale language model, but to investigate whether the proposed architectural mechanism provides a measurable advantage under small computational budgets.

## Experiment 1 — ~64K parameters

The first experiment compared SinGatedLM against a plain baseline.

### Seed 1337

| Model          | Parameters | Final validation loss |
| -------------- | ---------: | --------------------: |
| PlainLM        |     64,917 |                2.7491 |
| **SinGatedLM** | **64,918** |            **2.5604** |

SinGatedLM achieved:

$$
2.7491-2.5604=\boxed{0.1887}
$$

lower validation loss with essentially identical parameter counts.

### Seed 42

| Model          | Parameters | Final validation loss |
| -------------- | ---------: | --------------------: |
| PlainLM        |     64,917 |                2.7491 |
| **SinGatedLM** | **64,918** |            **2.6225** |

The advantage appeared again:

$$
2.7491-2.6225=\boxed{0.1266}
$$

### Convergence

In the seed-42 experiment, SinGatedLM reached a validation loss below the PlainLM's final result at approximately iteration 900:

```text
PlainLM final:        2.7491
SinGated @ 800:       2.7725
SinGated @ 900:       2.7299  ← already better
SinGated @ 2999:      2.6225
```

This suggests that the difference may involve not only final validation performance, but also **optimization/convergence behavior**.

---

# Experiment 2 — ~1M vs ~1.5M parameters

The architecture was then scaled substantially.

The SinGatedLM used approximately 1.03M parameters, while the PlainLM baseline used approximately 1.54M parameters.

### Results

| Model          |    Parameters | Final train loss | Final validation loss |      Time |
| -------------- | ------------: | ---------------: | --------------------: | --------: |
| **SinGatedLM** | **1,027,048** |       **1.6646** |            **1.8811** | **55.9s** |
| PlainLM        |     1,535,483 |           1.7996 |                2.0103 |     67.6s |

SinGatedLM achieved:

$$
\boxed{1.8811 < 2.0103}
$$

while using approximately:

$$
\frac{1,027,048}{1,535,483}\approx66.9\%
$$

of the baseline's parameters.

In other words, SinGatedLM used approximately **33% fewer parameters** while achieving lower validation loss in this experiment.

The validation-loss difference was:

$$
2.0103-1.8811=\boxed{0.1292}
$$

The model also reached a validation loss below the PlainLM's final result at approximately iteration 1900:

```text
PlainLM final:        2.0103

SinGated:
1800 → 2.0270
1900 → 1.9954  ← beats PlainLM final
2999 → 1.8811
```

---

# Current Results

Across the initial experiments:

```text
                 Validation Loss

64K
PlainLM          2.7491
SinGatedLM       2.5604
                 ↓ 0.1887

1M / 1.5M
PlainLM          2.0103
SinGatedLM       1.8811
                 ↓ 0.1292
```

The results are encouraging, but these experiments should be considered **initial evidence rather than a definitive architectural claim**.

The ~1M experiment is also not parameter-matched: SinGatedLM has ~1.03M parameters while the PlainLM has ~1.54M. The result therefore demonstrates an advantage under the tested configurations, but does not by itself establish superiority at an identical parameter budget.

---

# Why SinGated?

The motivation comes from treating neural transformations as more than repeated linear projections.

A conventional transformation might look like:

$$
y = Wx+b
$$

SinGated instead introduces a second variable:

$$
y=f(x,A)
$$

where the second variable is produced by attention:

$$
A=\operatorname{Attention}(x)
$$

This gives the network a learned context-dependent modulation mechanism:

$$
\boxed{
f(x,A)
=
(Wx+b)\odot\alpha\sin(A)
}
$$

The hypothesis is that this provides a different form of representational interaction from simply adding attention outputs to the residual stream.

---

# Experimental Philosophy

The project follows a simple principle:

> **Hypothesis → implementation → controlled experiment → measurement → scaling**

The purpose of the experiments is to determine whether the architectural mechanism actually provides useful behavior, rather than assuming that it does.

Future experiments can investigate:

* parameter-matched baselines
* additional random seeds
* different gating functions
* different values/parameterizations of \(\alpha\)
* removing the sinusoidal function
* replacing `sin` with other nonlinear functions
* larger and more diverse datasets
* language-model benchmarks
* training efficiency
* inference efficiency

---

# Status

🚧 **Experimental / Research**

Current results are promising, particularly because SinGatedLM has shown lower validation loss in both the ~64K experiments and the ~1M experiment.

The architecture is still being investigated and should not yet be interpreted as a generally superior replacement for standard Transformer components.

---

# License

MIT License

Copyright (c) 2026 bekhruzsuleyman

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

# Author

**Bekhruz Suleyman**
