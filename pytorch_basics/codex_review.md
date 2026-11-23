# Week 2 Review: PyTorch Tensors, Autograd, and Training

This document collects the eight assessment questions, worked answers, practical code, and important clarifications from our discussion. The assessment covered tensor shapes, broadcasting, automatic differentiation, gradient accumulation, manual parameter updates, and a small MLP trained on XOR.

The learning-rate measurements below are the results recorded during the assessment. Small numerical differences can occur across environments.

## 1. Tensor Dimensions and Multiplication

### Question

```python
import torch

x = torch.tensor([
    [1., 2., 3.],
    [4., 5., 6.]
])

w = torch.tensor([
    [1., 0.],
    [0., 1.],
    [1., 1.]
])
```

For each expression, determine whether it succeeds. Give its shape and values, or explain the error:

```python
a = x * x
b = x @ w
c = x @ x
d = x @ x.T
```

Explain the difference between `*` and `@`, and why fully connected layers use matrix multiplication.

### Answer

| Expression | Result | Shape |
| --- | --- | --- |
| `x * x` | `[[1, 4, 9], [16, 25, 36]]` | `(2, 3)` |
| `x @ w` | `[[4, 5], [10, 11]]` | `(2, 2)` |
| `x @ x` | Error: inner dimensions `3` and `2` do not match | — |
| `x @ x.T` | `[[14, 32], [32, 77]]` | `(2, 2)` |

`*` multiplies corresponding elements, subject to broadcasting. `@` performs matrix multiplication in this two-dimensional example.

A matrix with shape `(m, n)` can multiply one with shape `(n, p)`, producing `(m, p)`. A square matrix can multiply itself; `x @ x` fails here specifically because `x` is not appropriately shaped.

### Why Matrix Multiplication Represents a Layer

Each row of `x` is one sample. Each column of `w` contains one output neuron's weights for all input features:

| Input feature | Weight for neuron 1 | Weight for neuron 2 |
| --- | ---: | ---: |
| Feature 1 | 1 | 0 |
| Feature 2 | 0 | 1 |
| Feature 3 | 1 | 1 |

For sample `[1, 2, 3]`, the two neurons compute:

\[
z_1=1\times1+2\times0+3\times1=4,
\]

\[
z_2=1\times0+2\times1+3\times1=5.
\]

Matrix multiplication computes each input row's dot product with every weight column. Multiple weight columns therefore represent multiple output neurons, each receiving the same input features but combining them differently.

\[
X_{B\times D}\,W_{D\times H}=Z_{B\times H}
\]

Here, `B` is the number of samples, `D` is the number of input features, and `H` is the number of output neurons. A full layer can then add a bias and apply an activation:

\[
H=\operatorname{activation}(XW+b).
\]

## 2. Indexing and Broadcasting

### Question

```python
x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
featureBias = torch.tensor([10., 20., 30., 40.])
sampleOffset = torch.tensor([100., 200., 300.])
```

1. Give the shapes and values of `x[1]`, `x[1:2]`, `x[:, 1]`, and `x[:, 1:2]`.
2. Does `x + featureBias` succeed? Which values are added to each row?
3. Does `x + sampleOffset` succeed? Why?
4. Use broadcasting to add one offset to every element of its corresponding sample's row.

### Answer

| Expression | Shape | Values |
| --- | --- | --- |
| `x[1]` | `(4,)` | `[4, 5, 6, 7]` |
| `x[1:2]` | `(1, 4)` | `[[4, 5, 6, 7]]` |
| `x[:, 1]` | `(3,)` | `[1, 5, 9]` |
| `x[:, 1:2]` | `(3, 1)` | `[[1], [5], [9]]` |

Integer indexing removes the indexed axis. A slice such as `1:2` preserves it, with size `1`.

`x + featureBias` succeeds. Broadcasting aligns dimensions from the right:

```text
x:             (3, 4)
featureBias:      (4)
```

The result satisfies `result[i, j] = x[i, j] + featureBias[j]`. Since `featureBias` is one-dimensional, its element is indexed with `[j]`, not `[0][j]`.

`x + sampleOffset` fails: the trailing dimensions are `4` and `3`, which neither match nor equal `1`.

Add an axis to obtain a column-shaped offset:

```python
result = x + sampleOffset[:, None]
# Equivalent: x + sampleOffset.reshape(3, 1)
```

```text
x:               (3, 4)
sampleOffset:    (3, 1)
result:          (3, 4)
```

The result is:

```text
[[100, 101, 102, 103],
 [204, 205, 206, 207],
 [308, 309, 310, 311]]
```

Broadcasting does not generally require physically copying the smaller tensor into a larger one. Expansion is a useful way to reason about the values of the operation.

### Clarification: `(4)` Versus `(4,)`

This distinction comes from Python syntax:

```python
print(type((4)))   # <class 'int'>
print(type((4,)))  # <class 'tuple'>
```

The comma creates the one-element tuple. In a shape, each tuple element describes one axis:

| Shape | Number of axes | Meaning |
| --- | ---: | --- |
| `(4,)` | 1 | One axis of length 4 |
| `(1, 4)` | 2 | One row, four columns |
| `(4, 1)` | 2 | Four rows, one column |

The trailing comma in `(4,)` does not represent an extra empty axis.

### Clarification: The Type of `shape`

`tensor.shape` returns a `torch.Size`, which is a subclass of Python's `tuple`:

```python
x = torch.zeros(3, 4)

print(x.shape)                    # torch.Size([3, 4])
print(type(x.shape))              # <class 'torch.Size'>
print(isinstance(x.shape, tuple)) # True
print(tuple(x.shape))             # (3, 4)
print(x.shape[0])                 # 3
print(len(x.shape))               # 2
```

A shape needs an ordered collection because it records the size of every axis. A single total element count cannot distinguish shapes such as `(12,)`, `(3, 4)`, and `(2, 6)`. A `torch.Size` object is immutable; its printed square brackets do not make it a Python list.

## 3. Reshape Versus Transpose

### Question

```python
x = torch.arange(6).reshape(2, 3)
a = x.reshape(3, 2)
b = x.transpose(0, 1)
```

1. Give the shapes and values of `x`, `a`, and `b`.
2. Do `a` and `b` arrange the values identically?
3. If rows are samples and columns are features, does `a` preserve the original sample grouping?
4. Explain the difference between reshaping and transposing.

### Answer

```text
x, shape (2, 3):
[[0, 1, 2],
 [3, 4, 5]]

a, shape (3, 2):
[[0, 1],
 [2, 3],
 [4, 5]]

b, shape (3, 2):
[[0, 3],
 [1, 4],
 [2, 5]]
```

The output shapes match, but the arrangements differ.

For this example, `reshape` can be understood as flattening the values in their existing logical order and regrouping them into the new shape. This is a reasoning model, not a claim that an actual flattening copy always happens.

`transpose(0, 1)` exchanges two axes. For a matrix, this exchanges rows and columns; for a higher-dimensional tensor, it exchanges the two selected axes.

In `a`, the row `[2, 3]` mixes features from the two original samples. In `b`, each original sample is still grouped together, now in a column rather than a row. Transpose changes the location of the sample axis; reshape does not automatically preserve its meaning.

**Takeaway:** Matching shapes do not imply matching data arrangements or meanings. Choose the operation according to what each axis represents.

## 4. Autograd and Shared Variables

### Question

```python
x = torch.tensor([1., 2., 3.], requires_grad=True)
y = x * x + x
loss = y.sum()
```

1. Give the values and shapes of `y` and `loss`.
2. What is `x.grad` before `loss.backward()`?
3. What is `x.grad` afterward? Derive it.
4. Does `backward()` change the values in `x`?
5. What does `requires_grad=True` do, and how do the repeated uses of `x` contribute?

### Answer

`y` is `[2, 6, 12]`, with shape `(3,)`. `loss` is a scalar tensor containing `20`, with shape `()`.

Before the first backward pass, `x.grad` is `None`. Afterward:

```python
loss.backward()
print(x.grad)  # tensor([3., 5., 7.])
```

For each element:

\[
y_i=x_i^2+x_i,\qquad
\frac{\partial L}{\partial x_i}=x_i+x_i+1=2x_i+1.
\]

The two multiplication inputs each contribute `x[i]`, while the addition contributes `1`. These contributions sum separately for each element. They do not collapse the gradient vector into one number; `x.grad` has the same shape as `x`.

`backward()` computes and accumulates gradients. It does not update `x`.

In normal gradient mode, `requires_grad=True` enables recording the operations involving this leaf tensor so that backward can compute and accumulate its gradient. Newly created tensors normally default to `requires_grad=False`.

### Clarification: `None` Versus a Zero Gradient

- `x.grad is None`: no gradient tensor is currently stored.
- `x.grad` containing zeros: gradient storage exists, and its values are zero. This can result from computation or an explicit clearing operation.

Setting `requires_grad=True` does not immediately allocate a zero-filled `.grad` tensor.

### Clarification: Zero-Dimensional Tensors

A scalar tensor contains one value and has no axes:

| Construction | Shape | Dimensions | Elements |
| --- | --- | ---: | ---: |
| `torch.tensor(20.)` | `()` | 0 | 1 |
| `torch.tensor([20.])` | `(1,)` | 1 | 1 |
| `torch.tensor([[20.]])` | `(1, 1)` | 2 | 1 |

Dimension count and element count are different concepts.

```python
loss = torch.tensor([2., 6., 12.]).sum()
print(loss.shape)  # torch.Size([])
print(loss.ndim)   # 0
print(loss.numel())# 1
print(loss.item()) # 20.0, a Python number
```

There is no axis to index in a zero-dimensional tensor, so `loss[0]` is invalid. Keep the tensor itself for `backward()`; use `.item()` when you want a Python number for reporting.

## 5. Gradient Accumulation Across Forward Passes

### Question

Predict the three printed values and explain accumulation and clearing:

```python
w = torch.tensor(2.0, requires_grad=True)

lossA = w * w
lossA.backward()
print(w.grad.item())

lossB = w * w + w
lossB.backward()
print(w.grad.item())

w.grad.zero_()

lossC = w * w + w
lossC.backward()
print(w.grad.item())
```

Why does a fresh forward calculation still accumulate into `.grad`? How does this differ from accumulation along multiple paths within one graph?

### Answer

The outputs are `4.0`, `9.0`, and `5.0`.

\[
\frac{d(w^2)}{dw}=2w=4,\qquad
\frac{d(w^2+w)}{dw}=2w+1=5.
\]

The second backward call adds `5` to the existing `4`. Creating a fresh computation graph does not clear the leaf tensor's gradient buffer.

`w.grad.zero_()` clears the stored gradient without changing `w`. The next backward pass contributes `5`. Clearing once does not disable future accumulation.

- **Within one backward pass:** contributions through different paths sum to form the gradient of that loss.
- **Across backward passes:** new contributions accumulate into the same `.grad` buffer until it is cleared or replaced.

Accumulation across backward passes is not inherently wrong. Here, because `w` has not changed, the accumulated `9` equals the gradient of `lossA + lossB` at the same parameter value. If each update should use only the current loss, clear previous contributions before that loss's backward pass.

## 6. Debugging Manual Parameter Updates

### Question

```python
w = torch.tensor(0.0, requires_grad=True)
learningRate = 0.1

for step in range(3):
    loss = (w - 3.0) ** 2
    loss.backward()
    w -= learningRate * w.grad
```

1. Why does the update raise an error?
2. Fix the loop, including gradient clearing.
3. Explain `torch.no_grad()` and whether later computations can still track `w`.
4. Print the parameter, loss, and gradient with clear before/after timing.

### Answer

`w` is a leaf tensor requiring gradients. `-=` attempts to modify it in place while normal gradient tracking is enabled, which PyTorch prohibits for this operation.

Manual optimization should update the parameter without recording that update in the backward graph. Use `torch.no_grad()` for the update, and clear old gradients between optimization iterations.

```python
w = torch.tensor(0.0, requires_grad=True)
learningRate = 0.1

for step in range(3):
    wBefore = w.item()
    loss = (w - 3.0) ** 2
    loss.backward()
    gradient = w.grad.item()

    with torch.no_grad():
        w -= learningRate * w.grad

    # Clear after updating, preparing for the next iteration.
    w.grad.zero_()

    print(
        f"step={step}, wBefore={wBefore:.4f}, "
        f"lossBefore={loss.item():.4f}, gradient={gradient:.4f}, "
        f"wAfter={w.item():.4f}"
    )
```

| Step | Parameter before | Loss before | Gradient | Parameter after |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0 | 9 | -6 | 0.6 |
| 1 | 0.6 | 5.76 | -4.8 | 1.08 |
| 2 | 1.08 | 3.6864 | -3.84 | 1.464 |

### Clearing Before Versus After the Update

Both arrangements work for this exercise:

- Clear at the start of each iteration, guarding against `.grad` being `None` initially.
- Clear immediately after each update. Since backward has already populated gradients for these connected parameters, no initial `None` check is needed.

The gradient must still be available when updating, and old contributions must be cleared before the next backward pass if ordinary per-loss gradient descent is intended. Inspect or record gradients before clearing them.

### `no_grad()` Is Temporary

The context manager excludes operations inside its block from the backward graph. It does not permanently change `w.requires_grad` to `False`. The next forward pass, outside the block, constructs a new graph using the updated parameter.

Do not put the training forward calculation and loss inside `no_grad()` if you need to backpropagate through them.

## 7. Integrated Task: Train a Small MLP on XOR

### Question

Use these input-target pairs:

```text
[0, 0] -> 0
[1, 0] -> 1
[0, 1] -> 1
[1, 1] -> 0
```

Build a network with two input features, four hidden units, a `tanh` activation, and one output. Use mean squared error, automatic differentiation, and manual parameter updates.

1. Choose parameter shapes and annotate intermediate shapes.
2. Record the initial and final loss.
3. Verify that a weight tensor changes.
4. Explain why the hidden activation is needed.
5. Explain what goes wrong if targets have shape `(4,)` instead of `(4, 1)`.

### Answer: Shapes

The four samples are the batch dimension, not the number of input features.

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `inputs` | `(4, 2)` | Four samples, two features each |
| `firstWeight` | `(2, 4)` | Two input features to four hidden units |
| `firstBias` | `(4,)` | One bias per hidden unit |
| `hidden` | `(4, 4)` | Four hidden features per sample |
| `secondWeight` | `(4, 1)` | Four hidden features to one output |
| `secondBias` | `(1,)` | One output bias |
| `predictions` | `(4, 1)` | One prediction per sample |
| `targets` | `(4, 1)` | One target per sample |
| `loss` | `()` | Scalar mean squared error |

Each neuron shares its weights and bias across all samples. Biases broadcast across the batch dimension.

### Answer: Complete Implementation

```python
import torch

torch.manual_seed(42)

inputs = torch.tensor([
    [0., 0.],
    [1., 0.],
    [0., 1.],
    [1., 1.]
])
targets = torch.tensor([[0.], [1.], [1.], [0.]])

firstWeight = torch.randn(2, 4, requires_grad=True)
firstBias = torch.zeros(4, requires_grad=True)
secondWeight = torch.randn(4, 1, requires_grad=True)
secondBias = torch.zeros(1, requires_grad=True)

parameters = [firstWeight, firstBias, secondWeight, secondBias]

# Preserve independent copies before training for fair comparisons.
initialFirstWeight = firstWeight.detach().clone()
initialFirstBias = firstBias.detach().clone()
initialSecondWeight = secondWeight.detach().clone()
initialSecondBias = secondBias.detach().clone()


def forward(inputs):
    hidden = torch.tanh(inputs @ firstWeight + firstBias)
    predictions = hidden @ secondWeight + secondBias
    return predictions


learningRate = 0.1
numSteps = 2000

for step in range(numSteps):
    predictions = forward(inputs)
    loss = ((predictions - targets) ** 2).mean()
    loss.backward()

    with torch.no_grad():
        for parameter in parameters:
            parameter -= learningRate * parameter.grad

    for parameter in parameters:
        parameter.grad.zero_()

    if step % 200 == 0:
        print(f"step={step}, lossBeforeUpdate={loss.item():.6f}")

with torch.no_grad():
    finalPredictions = forward(inputs)
    finalLoss = ((finalPredictions - targets) ** 2).mean()
    maxChange = (firstWeight - initialFirstWeight).abs().max().item()

print("Final predictions:")
print(finalPredictions)
print(f"Final loss: {finalLoss.item():.12e}")
print(f"Maximum first-layer weight change: {maxChange:.12e}")
```

Recorded predictions with learning rate `0.1` were approximately:

```text
[[5.0478e-07],
 [1.0000e+00],
 [1.0000e+00],
 [1.1008e-06]]
```

The network fitted the four XOR examples. A printed loss of `0.000000` rounds to six decimal places; it does not establish an exactly zero loss. Scientific notation exposes the remaining error.

### Why Nonlinearity Matters

Without an activation, the two affine layers collapse into one:

\[
(XW_1+b_1)W_2+b_2
=X(W_1W_2)+(b_1W_2+b_2).
\]

An affine output has a fixed coefficient for each feature. XOR instead requires the effect of one feature to depend on the other:

- If `x2 = 0`, changing `x1` from `0` to `1` should increase the output.
- If `x2 = 1`, the same change should decrease the output.

For a single affine function `y = w1*x1 + w2*x2 + b`, the first three XOR samples require `b = 0`, `w1 = 1`, and `w2 = 1`. The fourth then produces `2`, not `0`. No such affine function fits all four targets exactly.

For a hand-computable nonlinear example, let `ReLU(z) = max(0, z)` and define:

\[
h_1=\operatorname{ReLU}(x_1+x_2),\qquad
h_2=\operatorname{ReLU}(x_1+x_2-1),\qquad
y=h_1-2h_2.
\]

| Input | `h1` | `h2` | Output |
| --- | ---: | ---: | ---: |
| `(0, 0)` | 0 | 0 | 0 |
| `(1, 0)` | 1 | 0 | 1 |
| `(0, 1)` | 1 | 0 | 1 |
| `(1, 1)` | 2 | 1 | 0 |

The second hidden unit provides a correction only when the input sum exceeds `1`. Nonlinear hidden units can thus create useful intermediate features that a final affine layer combines.

The actual network uses `tanh`, a smooth nonlinear activation. Different weights and biases place its sensitive and saturated regions at different inputs. Combining several such responses allows the network to represent the required relationship. The ReLU construction is an illustration, not a claim about the exact features learned by the trained `tanh` network.

### The Silent Target-Shape Bug

If predictions have shape `(4, 1)` but targets have shape `(4,)`, subtraction broadcasts to `(4, 4)`:

\[
error_{ij}=prediction_i-target_j.
\]

This compares every prediction against every target, rather than its corresponding target. The code can run and produce a scalar loss while optimizing the wrong objective.

Keep both tensors shaped `(4, 1)` so subtraction produces four corresponding errors.

## 8. Learning Rate and Evidence-Based Debugging

### Question

Compare two learning rates differing by a factor of ten. Start from identical parameter values and use the same number of updates.

1. Record the initial, intermediate, and final loss.
2. Explain the differences between the loss curves.
3. Explain why identical initialization matters.
4. Give two possible causes of constant loss and a concrete check for each.
5. What would you inspect if loss grows rapidly or becomes `NaN`?
6. Does a nonzero gradient prove that the parameter changed? How can you verify an update directly?

### Answer: Restoring the Same Initialization

Before the second experiment, restore all four parameters from the snapshots saved before the first experiment:

```python
firstWeight = initialFirstWeight.detach().clone().requires_grad_(True)
firstBias = initialFirstBias.detach().clone().requires_grad_(True)
secondWeight = initialSecondWeight.detach().clone().requires_grad_(True)
secondBias = initialSecondBias.detach().clone().requires_grad_(True)

# Rebuild the list so updates target the newly restored tensors.
parameters = [firstWeight, firstBias, secondWeight, secondBias]
learningRate = 0.01
```

Then repeat the training and evaluation portion of Question 7. Do not overwrite the original snapshots with already-trained parameters.

- `detach()` separates the tensor from its prior computation history.
- `clone()` creates independent storage, protecting the saved initialization from in-place training updates.
- `requires_grad_(True)` enables gradient tracking on the new leaf tensor and returns that tensor, allowing chaining.

After `detach().clone()`, gradient tracking must be explicitly re-enabled. It only needs to be enabled once when restoring parameters, not once per training iteration. Creating a tensor with `requires_grad=True` already enables it.

Simply continuing training from the first experiment's final weights would compare different starting points, confounding the effect of learning rate.

### Recorded Comparison

Intermediate losses were logged before the update with the indicated zero-based step number; final losses were recomputed after all 2,000 updates.

| Measurement | Learning rate `0.1` | Learning rate `0.01` |
| --- | ---: | ---: |
| Initial loss | 0.527647 | 0.527647 |
| Step 200, before update | 0.008351 | 0.190523 |
| Step 400, before update | 0.000008 | 0.159861 |
| Step 600, before update | 0.000000, rounded | 0.135036 |
| Step 1000, before update | 0.000000, rounded | 0.086391 |
| Step 1800, before update | 0.000000, rounded | 0.014669 |
| After 2000 updates | Near zero | 0.008156 |

Recorded predictions with learning rate `0.01` were approximately:

```text
[[0.0532],
 [0.9325],
 [0.9016],
 [0.1248]]
```

Both rates learned, but `0.1` converged faster for this initialization, architecture, and training budget. The smaller rate after 2,000 updates reached roughly the loss of the larger rate around step 200. This observation is specific to this experiment; increasing the learning rate indefinitely would not guarantee faster convergence.

### Debugging Constant Loss

| Possible cause | Concrete check |
| --- | --- |
| All gradients are zero, giving zero updates in this manual gradient-descent loop | Inspect gradients immediately after backward and before clearing |
| Gradients exist, but the parameter update is missing or does not modify the parameters used in forward | Compare independent parameter snapshots immediately before and after the update |

Inspecting `.grad` after `zero_()` cannot establish that backward produced zero gradients. Also increase displayed loss precision before concluding that the loss is exactly constant.

If parameters change but the displayed loss does not, check that a fresh forward pass and loss are computed each iteration, rather than printing a stale loss.

### Debugging Increasing Loss or `NaN`

First check the update sign:

```python
parameter -= learningRate * parameter.grad
```

Using `+=` instead follows the gradient uphill locally. Next, check whether the learning rate is too large: even a correct descent direction can overshoot and cause oscillation or divergence.

If those checks do not resolve the issue, locate where the first non-finite value appears: forward outputs, loss, gradients, or updated parameters. Invalid arithmetic such as division by zero can also cause non-finite values. `torch.isfinite()` can check for both infinities and `NaN`.

### Verifying That Parameters Changed

A nonzero gradient does not prove an update happened. Backward and parameter updates are separate operations.

Comparing a parameter before consecutive forward passes can reveal a change. A more direct check compares it around the update itself:

```python
before = parameter.detach().clone()

with torch.no_grad():
    parameter -= learningRate * parameter.grad

maxChange = (parameter.detach() - before).abs().max().item()
print(f"Maximum parameter change: {maxChange:.12e}")
```

A positive `maxChange` establishes that at least one element changed. Saving `before = parameter` would only create another reference to the same tensor, not preserve its earlier values. `detach()` alone also shares storage, so the independent copy is important.

## Assessment Outcome and Next Step

The review and training exercise established the core Week 2 skills:

- Distinguishing sample count, feature count, and neuron count.
- Predicting tensor shapes and interpreting broadcasting.
- Preserving axis meanings when reshaping or transposing.
- Computing and explaining gradients with autograd.
- Managing accumulation within and across backward passes.
- Separating gradient computation from manual parameter updates.
- Training a nonlinear MLP and comparing learning rates fairly.
- Checking numerical behavior and verifying parameter changes directly.

Week 2 is complete. Week 3 introduces `Dataset`, `DataLoader`, and `nn.Module`, using them to organize a full classification project.

## Official References

- [PyTorch: Tensors](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
- [PyTorch: torch.Size](https://docs.pytorch.org/docs/stable/size.html)
- [PyTorch: Automatic Differentiation](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
- [PyTorch: Optimizing Model Parameters](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
- [PyTorch: Autograd Mechanics and No-Grad Mode](https://docs.pytorch.org/docs/stable/notes/autograd.html#no-grad-mode)
