# Simple Linear Regression

## What Are We Trying To Do?

Given data points — pairs of $(x_i, y_i)$ — we want to find a straight line that **best fits** them:

$$\hat{y} = wx + b$$

Where:
- $w$ = weight (slope)
- $b$ = bias (intercept)
- $\hat{y}$ = our prediction

---

## Part 1 — Closed Form Solution (Ordinary Least Squares)

### Defining "Error"

For each data point, our prediction is off by some amount — the **residual**:

$$e_i = y_i - \hat{y}_i = y_i - (wx_i + b)$$

We can't just sum residuals — positive and negative errors cancel out. So we **square** them:

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - wx_i - b)^2$$

This is our **loss function**. This approach is called **Ordinary Least Squares (OLS)**.

---

### Minimizing the Loss — Deriving $b$

A function is at its minimum where its **derivative = 0**. We use the **chain rule**:

$$\frac{d}{db}(u)^2 = 2u \cdot \frac{du}{db}$$

Where $u = y_i - wx_i - b$. Differentiating $u$ term by term:

- $\frac{d}{db}(y_i) = 0$ → constant
- $\frac{d}{db}(wx_i) = 0$ → constant
- $\frac{d}{db}(-b) = -1$ → **this gives us the -1**

So by chain rule:

$$\frac{d}{db}(u)^2 = 2(y_i - wx_i - b) \cdot (-1) = -2(y_i - wx_i - b)$$

> 💡 **Discovery:** The **-2** comes from two places: the **2** from the power rule, and the **-1** from differentiating $-b$.

Applying to the full MSE and setting to zero:

$$\frac{-2}{n} \sum_{i=1}^{n} (y_i - wx_i - b) = 0$$

Multiply both sides by $\frac{n}{-2}$:

$$\sum_{i=1}^{n} (y_i - wx_i - b) = 0$$

Rearranging gives us:

$$\boxed{b = \bar{y} - w\bar{x}}$$

> 💡 **Discovery:** The line **always passes through the mean point** $(\bar{x}, \bar{y})$. Once we know $w$, the intercept $b$ is just whatever shifts the line to pass through the center of gravity of the data.

---

### Minimizing the Loss — Deriving $w$

Differentiating MSE with respect to $w$. Here $u = y_i - wx_i - b$, and:

- $\frac{d}{dw}(y_i) = 0$
- $\frac{d}{dw}(-wx_i) = -x_i$ → **note the negative sign stays!**
- $\frac{d}{dw}(-b) = 0$

So $\frac{du}{dw} = -x_i$, and by chain rule:

$$\frac{d}{dw}(u)^2 = 2(y_i - wx_i - b) \cdot (-x_i) = -2x_i(y_i - wx_i - b)$$

> 💡 **Discovery:** A common mistake is writing $+x_i$ here. The term is $-wx_i$, so differentiating removes $w$ but **keeps the negative**, giving $-x_i$.

Applying to the full MSE and setting to zero:

$$\frac{-2}{n} \sum_{i=1}^{n} x_i(y_i - wx_i - b) = 0 \implies \sum_{i=1}^{n} x_i(y_i - wx_i - b) = 0$$

Substituting $b = \bar{y} - w\bar{x}$ and simplifying:

$$\sum_{i=1}^{n} x_i\big((y_i - \bar{y}) - w(x_i - \bar{x})\big) = 0$$

Expanding:

$$\sum_{i=1}^{n} x_i(y_i - \bar{y}) = w\sum_{i=1}^{n} x_i(x_i - \bar{x})$$

Using the identity $\sum x_i(y_i - \bar{y}) = \sum (x_i - \bar{x})(y_i - \bar{y})$ — which holds because the sum of deviations from the mean is always zero — we get:

$$\boxed{w = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n}(x_i - \bar{x})^2}}$$

---

### Intuition Behind the Formula

$$w = \frac{\text{covariance of } x \text{ and } y}{\text{variance of } x}$$

The numerator measures **how much x and y move together**. The denominator measures **how much x varies on its own**.

Think of it as: *"For every unit that x spreads out, how many units does y move with it?"*

- x and y move together a lot, x doesn't spread much → **steep slope**
- x barely affects y, but x spreads widely → **shallow slope**

---

## Part 2 — Gradient Descent

Instead of the closed-form solution, gradient descent **iteratively** finds $w$ and $b$ by taking small steps in the direction that reduces loss.

### The Update Rule

At each step:

$$w \leftarrow w - \alpha \cdot \frac{\partial \text{MSE}}{\partial w}$$

$$b \leftarrow b - \alpha \cdot \frac{\partial \text{MSE}}{\partial b}$$

Where $\alpha$ is the **learning rate**.

The gradients are:

$$\frac{\partial \text{MSE}}{\partial w} = \frac{-2}{n} \sum_{i=1}^{n} x_i(y_i - \hat{y}_i)$$

$$\frac{\partial \text{MSE}}{\partial b} = \frac{-2}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)$$

---

### Why the Weight Gradient is Larger Than the Bias Gradient

Notice the **key difference** between the two gradients:

- $\frac{\partial \text{MSE}}{\partial w}$ has $x_i$ multiplied inside the sum
- $\frac{\partial \text{MSE}}{\partial b}$ does **not**

With the same learning rate, weight gets updates ~7x larger than bias when x values are large (e.g., x in [1, 15]).

> 💡 **Discovery:** This is why $b$ converges much slower than $w$ when x values are large. The $x_i$ multiplier acts as an **amplifier** on weight gradients.

---

## Part 3 — Feature Scaling

### Min-Max Normalization

To fix the gradient imbalance, we scale x values to the range [0, 1]:

$$x_{scaled} = \frac{x_i - x_{min}}{x_{max} - x_{min}}$$

Now $x_i \in [0, 1]$, so the weight gradient is no longer amplified. Both $w$ and $b$ receive proportionally similar gradient updates and converge at a similar rate.

### Why We Don't Scale Y

Y appears only inside the residual $(y_i - \hat{y}_i)$. It is never multiplied by itself as a standalone scaling factor like $x_i$ is. So Y doesn't cause the same amplification problem.

> 💡 **Note:** You would scale Y when its values are very large (e.g., house prices in millions), where large residuals would blow up gradients.

---

### Critical Discovery: Scaling Changes the Weight Interpretation

After scaling x to [0, 1], the weight $w$ is **no longer interpretable** as the original slope. It becomes much larger because now a tiny change in scaled x must produce a large change in y.

Example — true relationship `y = 0.5x`:

| | $w$ | $b$ |
|---|---|---|
| Unscaled | ~0.5 | ~0.0 |
| Scaled | ~28.0 | ~3.0 |

Both are correct — they describe the **same line** in different coordinate spaces. Predictions are what matter, not the raw weights.

---

### Critical Discovery: When Scaling Hurts Extrapolation

> 💡 **Discovery:** Scaling is not always better. It can actually hurt predictions when test data is far outside the training range.

**Scenario:**

```
train_x = [1, 2, 4, 4, 5, 6, 7]   # max = 7
test_x  = [137]                     # far outside training range
```

After scaling with training min=1, max=7:

```
scaled x=137 → (137-1)/(7-1) = 22.67
```

The model was trained only on values between 0 and 1. Asking it to predict at 22.67 is **massive extrapolation in scaled space**, which causes large prediction errors.

The **unscaled model** doesn't have this problem — it learned the relationship directly in the original space where x=137 is a natural extension.

**Rule of thumb:**
- Use scaling when train and test data live in the **same range**
- Be cautious with scaling when test data extends **far beyond** training range
- Or scale using a range that covers **both** train and test data

---

### Critical Discovery: Scaling Must Use Training Statistics

When scaling test data, **always use the training data's min and max** — never recompute from the test set.

```python
# Correct
x_train_scaled, min_x, max_x = scaler(x_train)
x_test_scaled, _, _ = scaler(x_test, min_x, max_x)  # use training stats

# Wrong
x_test_scaled, _, _ = scaler(x_test)  # uses test's own min/max
```

In real deployment, you never know the test min/max in advance. Scaling test data with its own statistics means the model sees values on a **different scale** than it was trained on.

---

## Part 4 — Convergence and Hyperparameters

### Epochs vs Learning Rate

| Hyperparameter | Effect |
|---|---|
| More epochs | More time to converge, but slower |
| Higher learning rate | Faster convergence, but risk of overshooting |
| Too high learning rate | Divergence — loss goes up instead of down |

> 💡 **Discovery:** You cannot compensate for a slow $b$ convergence by simply boosting the learning rate. Since both $w$ and $b$ share the same rate, boosting it would cause $w$'s massive gradient updates to **overshoot**, potentially causing divergence.

---

### Extrapolation Works for Truly Linear Data

> 💡 **Discovery:** Linear regression finds a line — and a line is infinite. Once the model has found the true line, it can correctly predict for **any** value of x, not just values inside the training range.

The caveat: this only works once the model has **converged**. With too few epochs, the model hasn't found the true line yet, so extrapolated predictions will be off — not because extrapolation is wrong, but because the line itself is wrong.

---

## Part 5 — Evaluation Metrics

| Metric | Formula | Notes |
|---|---|---|
| MSE | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | Penalizes large errors heavily |
| RMSE | $\sqrt{\text{MSE}}$ | Same units as y, more interpretable |
| MAE | $\frac{1}{n}\sum\|y_i - \hat{y}_i\|$ | More robust to outliers |
| R² | $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ | 1.0 = perfect fit, 0 = no better than mean |

---


## Implementation Files

| File | Description |
|---|---|
| `simple_linear_regression.py` | Full class-based implementation — closed form + gradient descent |
| `simple_linear_regression.ipynb` | Walkthrough notebook with plots and experiments |