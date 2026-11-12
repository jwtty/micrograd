# Week 1 Review: Derivatives, Computation Graphs, and Backpropagation

This review contains eight questions and worked answers from the first week of the neural-network foundations study plan. It focuses on scalar derivatives, the chain rule, gradient accumulation, reverse topological order, numerical gradient checking, and gradient descent.

The exercises do not require PyTorch or a complete neural network implementation. Try each question before reading its answer.

## 1. What Does a Gradient Mean?

### Question

Suppose the loss is \(L(w)\). At \(w=2\), its derivative is:

\[
\frac{dL}{dw}=5.
\]

1. What does the value `5` mean?
2. What happens to the loss if you increase \(w\) by a small positive amount?
3. Which direction should you move \(w\) to decrease the loss?
4. Does this guarantee that decreasing \(w\) by `10` will decrease the loss? Why?

### Answer

The derivative describes the local rate of change of the loss with respect to \(w\). For a sufficiently small change \(\Delta w\):

\[
L(2+\Delta w)-L(2)\approx 5\Delta w.
\]

1. A small change in \(w\) produces approximately five times that change in the loss, with the sign included.
2. A small positive change in \(w\) increases the loss by approximately \(5\Delta w\).
3. A sufficiently small move in the negative direction decreases the loss.
4. No. The derivative is local information. It does not guarantee what happens after a large move such as `-10`.

**Key point:** A descent direction and a suitable step size are separate requirements. A step can move in the right direction initially but be too large to decrease the loss.

## 2. Draw a Computation Graph and Apply the Chain Rule

### Question

Given:

\[
u=ab+c,\qquad L=u^2,
\]

with \(a=2\), \(b=3\), and \(c=-4\):

1. Draw the computation graph and calculate all forward values.
2. Calculate \(\partial L/\partial u\), \(\partial L/\partial a\), \(\partial L/\partial b\), and \(\partial L/\partial c\).
3. Explain one parameter's gradient as a product of local derivatives.

### Answer

Introduce an intermediate value \(p=ab\):

```text
a = 2 ──┐
        ├── multiply ──> p = 6 ──┐
b = 3 ──┘                       ├── add ──> u = 2 ── square ──> L = 4
c = -4 ─────────────────────────┘
```

The forward values are:

\[
p=2\times3=6,\qquad u=6-4=2,\qquad L=2^2=4.
\]

Start the backward pass with \(\partial L/\partial L=1\). Then:

\[
\frac{\partial L}{\partial u}=2u=4,
\]

\[
\frac{\partial L}{\partial a}
=\frac{\partial L}{\partial u}\frac{\partial u}{\partial a}
=4\times b=12,
\]

\[
\frac{\partial L}{\partial b}
=\frac{\partial L}{\partial u}\frac{\partial u}{\partial b}
=4\times a=8,
\]

\[
\frac{\partial L}{\partial c}
=\frac{\partial L}{\partial u}\frac{\partial u}{\partial c}
=4\times1=4.
\]

For example, changing \(a\) affects \(p\), then \(u\), then \(L\). The local derivatives along that path are multiplied:

\[
\frac{\partial L}{\partial a}
=\frac{\partial L}{\partial u}
\frac{\partial u}{\partial p}
\frac{\partial p}{\partial a}
=4\times1\times3=12.
\]

**Implementation connection:** Setting `L.grad = 1` seeds backpropagation because the derivative of the loss with respect to itself is `1`.

## 3. What Happens When a Variable Is Used More Than Once?

### Question

Given:

\[
b=a\times a,\qquad L=b+a,\qquad a=3:
\]

1. What is \(dL/da\)?
2. Through which paths does \(a\) affect \(L\)?
3. What goes wrong if each backward contribution uses `a.grad = contribution`?
4. Why is `a.grad += contribution` necessary?

### Answer

Since \(L=a^2+a\):

\[
\frac{dL}{da}=2a+1=7.
\]

There are two broad routes: through \(b\), and directly through the addition in \(L=b+a\). Expanding the multiplication reveals three individual contributions:

| Use of `a` | Contribution to `a.grad` |
| --- | ---: |
| Direct input to `L = b + a` | 1 |
| Left input to `b = a * a` | 3 |
| Right input to `b = a * a` | 3 |
| **Total** | **7** |

Using assignment overwrites an earlier contribution. The result may then depend on the traversal order and fail to include every path.

Using `+=` adds the contributions together, correctly accounting for the total effect of changing \(a\).

**Key point:** Accumulation is necessary both when a variable is reused across operations and when both inputs of one operation reference the same object.

## 4. Implement Backward Rules for Addition and Multiplication

### Question

Assume all `.grad` fields start at `0`, and the output has already received its gradient from downstream operations. Complete these rules:

```python
# out = x + y
def backwardAdd():
    x.grad += __________
    y.grad += __________

# out = x * y
def backwardMul():
    x.grad += __________
    y.grad += __________
```

Why is `out.grad` needed? Does the multiplication rule remain correct when `x` and `y` reference the same object?

### Answer

```python
# out = x + y
def backwardAdd():
    x.grad += out.grad
    y.grad += out.grad

# out = x * y
def backwardMul():
    x.grad += y.data * out.grad
    y.grad += x.data * out.grad
```

These are closures: `x`, `y`, and `out` refer to the nodes for the particular operation.

For multiplication, the chain rule gives:

\[
\frac{\partial L}{\partial x}
=\frac{\partial L}{\partial out}
\frac{\partial out}{\partial x}
=out.grad\times y.data.
\]

`out.grad` represents the derivative of the **final loss** with respect to `out`. Multiplying it by the local derivative extends that information backward through the current operation.

When `x` and `y` are the same object, both statements update the same `.grad` field. The operation therefore contributes:

\[
2\times value.data\times out.grad.
\]

**Key point:** A node's `.grad` represents the derivative of the final loss with respect to that node, not merely the derivative of the next operation.

## 5. Why Must Backpropagation Follow a Particular Order?

### Question

Given:

```python
p = a * b
q = p + c
L = q * p
```

1. In what order should `L`, `q`, and `p` execute their backward rules?
2. Why must `p` wait until `q` has propagated its contribution?
3. Why does topological sorting use a `visited` set?
4. Does processing each node once contradict accumulating multiple gradient contributions?

### Answer

The backward order is:

```text
L → q → p
```

Initialize gradients to `0` and seed `L.grad = 1`.

**First, process `L = q * p`:**

```python
q.grad += p.data * L.grad
p.grad += q.data * L.grad
```

At this point, `q.grad` is complete, but `p.grad` has received only the direct contribution from `L`.

**Next, process `q = p + c`:**

```python
p.grad += q.grad
c.grad += q.grad
```

Now `p.grad` has received both contributions:

\[
\frac{\partial L}{\partial p}=q+p.
\]

**Finally, process `p = a * b`:**

```python
a.grad += b.data * p.grad
b.grad += a.data * p.grad
```

Processing `p` too early would propagate an incomplete gradient to `a` and `b`.

The `visited` set ensures a shared node appears only once in the topological list. Otherwise its backward rule could run repeatedly and propagate contributions more than once.

There is no contradiction:

- A node can **receive multiple contributions** from operations that use it.
- It executes its own backward rule **once per backward pass**, after receiving all those contributions.

**Key point:** Reverse topological order ensures each operation propagates its complete accumulated gradient to its inputs.

## 6. How Can You Check a Gradient Without Automatic Differentiation?

### Question

Use the same function and values as Question 2:

```python
def loss(a, b, c):
    return (a * b + c) ** 2

a = 2.0
b = 3.0
c = -4.0
```

Estimate the partial derivative with respect to `a` using central differences:

\[
\frac{\partial L}{\partial a}
\approx
\frac{L(a+h,b,c)-L(a-h,b,c)}{2h}.
\]

1. Why must `b` and `c` remain fixed?
2. Try `h = 1e-1`, `1e-3`, `1e-6`, and `1e-12`.
3. Is a smaller `h` always more accurate?

### Answer

A partial derivative measures how the function changes with one variable while the others remain fixed. Changing `b` or `c` would mix their effects into the estimate.

Runnable implementation:

```python
def loss(a, b, c):
    return (a * b + c) ** 2


def finiteDifference(h, a, b, c):
    if h <= 0:
        raise ValueError("h must be positive")
    return (loss(a + h, b, c) - loss(a - h, b, c)) / (2.0 * h)


a, b, c = 2.0, 3.0, -4.0
exactGradient = 12.0

for h in (1e-1, 1e-3, 1e-6, 1e-12):
    estimate = finiteDifference(h, a, b, c)
    absoluteError = abs(estimate - exactGradient)
    print(f"h={h:.0e}, gradient={estimate:.15g}, error={absoluteError:.6g}")
```

Results observed with Python floating-point arithmetic:

| `h` | Estimated gradient | Absolute error from 12 |
| --- | ---: | ---: |
| `1e-1` | 12.0000000000000 | approximately `3.02e-14` |
| `1e-3` | 12.0000000000007 | approximately `6.77e-13` |
| `1e-6` | 11.9999999999010 | approximately `9.90e-11` |
| `1e-12` | 12.0010668069881 | approximately `1.07e-3` |

In general, reducing `h` reduces the approximation error of central differences for sufficiently smooth functions. However, very small `h` makes floating-point rounding and cancellation important. Subtracting nearly equal function values and dividing by a tiny number can magnify numerical error.

This particular function is quadratic in `a`. In exact arithmetic, central differences are already exact for every nonzero `h`:

\[
\frac{(2+3h)^2-(2-3h)^2}{2h}=12.
\]

Thus, the errors in this example come from floating-point computation rather than a central-difference approximation error.

**Python pitfall from the exercise:** Calculating an expression inside a function does not return it automatically. Without `return`, the function returns `None`:

```python
def derivativeWithoutReturn(h, a, b, c):
    (loss(a + h, b, c) - loss(a - h, b, c)) / (2.0 * h)


assert derivativeWithoutReturn(0.1, a, b, c) is None
```

## 7. How Does a Gradient Become a Parameter Update?

### Question

Given:

\[
L(w)=(w-3)^2,\qquad w_0=0,
\]

perform two gradient-descent updates with learning rate \(\eta=0.1\).

For each update, list the current parameter, current loss, gradient, and updated parameter. Should the second step reuse or accumulate the first step's gradient?

### Answer

The derivative and update rule are:

\[
\frac{dL}{dw}=2(w-3),\qquad
w_{t+1}=w_t-0.1\times2(w_t-3).
\]

| Update | Current `w` | Current loss | `dL/dw` | Updated `w` |
| --- | ---: | ---: | ---: | ---: |
| First | 0 | 9 | -6 | 0.6 |
| Second | 0.6 | 5.76 | -4.8 | 1.08 |

After the second update, the loss is:

\[
L(1.08)=(1.08-3)^2=3.6864.
\]

For this ordinary gradient-descent procedure, recompute the gradient at the new parameter value on every iteration. When implementing it with gradient buffers, clear the previous gradients before the next backward pass.

Two different uses of accumulation must be distinguished:

- **Within one backward pass:** Add contributions from every path leading to the loss.
- **Between these optimization iterations:** Clear old gradients, run a new forward pass, and compute fresh gradients at the updated parameters.

Also, `backward()` computes gradients; the parameter-update step changes parameter values. They are different operations.

## 8. Does a Zero Gradient Mean the Loss Is Minimized?

### Question

Given:

\[
L(w)=(w^2-1)^2:
\]

1. What is the gradient at `w = 0`?
2. Will ordinary gradient descent update `w` from that exact point?
3. Is the loss minimized there?
4. What does a zero gradient establish, and what does it not guarantee?

### Answer

Differentiate using the chain rule:

\[
L'(w)=2(w^2-1)\times2w=4w(w^2-1).
\]

At `w = 0`, the gradient is `0`, so ordinary gradient descent will not move from that exact point. The loss is `1` there, and this point is a local maximum.

Both `w = 1` and `w = -1` give loss `0`. Since a square cannot be negative, these are global minima.

More generally, a zero gradient identifies a **stationary point**, not necessarily a maximum or minimum. For example:

\[
f(w)=w^3,\qquad f'(0)=0.
\]

Values immediately to the left of `0` are smaller, while values immediately to the right are larger. This stationary point is neither a local maximum nor a local minimum. In multiple dimensions, a stationary point can also be a saddle point.

**Key point:** A zero gradient alone does not establish local or global optimality. It also does not imply that the function is constant in a neighborhood.

## Takeaways and Readiness for Week 2

The discussion established the following conceptual foundations:

- A derivative describes local sensitivity, not the effect of an arbitrary large move.
- Backpropagation multiplies local derivatives along a path and sums contributions across paths.
- Gradient accumulation must handle shared nodes and repeated inputs correctly.
- Each operation executes its backward rule once, after its output gradient is complete.
- Numerical differences provide an independent gradient check, subject to floating-point limitations.
- Gradient computation and parameter updates are separate steps.
- Fresh optimization iterations generally require fresh gradients at the new parameter values.
- A zero gradient does not guarantee a minimum.

The eight-question conceptual review is complete. The next step is Week 2: reproduce these calculations with PyTorch tensors and automatic differentiation, compare `.grad` with hand calculations, and train a small network while tracking tensor shapes and parameter updates.
