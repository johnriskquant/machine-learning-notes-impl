## Week 2: Multiple Linear Regression & Gradient Descent in Practice

### 1. Multiple Linear Regression
In univariate linear regression, we had a single feature: $f_{w,b}(x) = wx + b$. In multiple linear regression, we use multiple features to predict the target.

**Notation:**
*   $x_j$ = the $j$-th feature
*   $n$ = total number of features
*   $\vec{x}^{(i)}$ = features of the $i$-th training example (represented as a row vector)
*   $x_j^{(i)}$ = value of feature $j$ in the $i$-th training example

**Model Function:**
For $n$ features, the model expands to:
$$f_{w,b}(\vec{x}) = w_1x_1 + w_2x_2 + \dots + w_nx_n + b$$
We can represent the parameters and features as vectors:
*   $\vec{w} = [w_1, w_2, \dots, w_n]$ 
*   $\vec{x} = [x_1, x_2, \dots, x_n]$
*   $b$ is a scalar number.

Using the dot product, the model is written concisely as:
$$f_{\vec{w},b}(\vec{x}) = \vec{w} \cdot \vec{x} + b$$

---

### 2. Vectorization
Vectorization makes code shorter, more readable, and significantly faster by taking advantage of linear algebra libraries (like NumPy) and parallel processing hardware (GPUs/modern CPUs).

**Without Vectorization (Slow):**
```python
f = 0
for j in range(0, n):
    f = f + w[j] * x[j]
f = f + b
```
**With Vectorization (Fast):**
```
f = np.dot(w, x) + b
```
Why is it faster? `np.dot()` grabs all values of w and x and multiplies them in parallel rather than sequentially, turning calculations that could take hours into minutes.

### 3. Gradient Descent for Multiple Linear Regression
To train the model, we use gradient descent to simultaneously update all parameters $w_1 \dots w_n$ and $b$.Repeat until convergence:
w_j = w_j - \alpha \frac{\partial}{\partial w_j} J(\vec{w},b) \quad \text{for } j = 1 \dots n
b = b - \alpha \frac{\partial}{\partial b} J(\vec{w},b)
w_j = w_j - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)} \right) x_j^{(i)}
b = b - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)} \right)

### 4. Alternative: The Normal Equation
* An analytical way to solve for $w$ and $b$ exactly without needing iterations
* Drawback: It only works for linear regression (does not generalize to other algorithms like logistic regression or neural networks) and becomes exceedingly slow when the number of features is large ($n > 10,000$). Gradient descent is the recommended standard.

### 5. Feature Scaling
When features have drastically different ranges (e.g., number of bedrooms vs. square footage), the cost function $J$ becomes highly sensitive to small features and insensitive to large ones. 

*   **The Problem:** This asymmetry stretches the cost function into an elongated, elliptical valley. Gradient descent gets trapped zig-zagging slowly across the steep walls rather than moving directly to the global minimum.
*   **The Solution (Scaling):** Rescaling features transforms the cost function contours into concentric circles, allowing the gradient to point directly at the minimum for much faster convergence.
*   **Guideline:** Aim to scale every feature $x_j$ so it ranges roughly between $-1 \le x_j \le 1$.

**Common Scaling Approaches:**
1.  **Divide by Maximum:** $x_{scaled} = \frac{x}{x_{max}}$
2.  **Mean Normalization:** Center data around 0. $x_{scaled} = \frac{x - \mu}{\max - \min}$
3.  **Z-score Normalization:** $x_{scaled} = \frac{x - \mu}{\sigma}$

---

### 6. Checking Convergence & Choosing a Learning Rate ($\alpha$)
Plot the cost function $J(\vec{w},b)$ against the number of iterations to create a **Learning Curve**.

*   **Checking Convergence:** The curve should be strictly monotonic (decreasing after every single iteration). You can use an automatic convergence test: declare convergence if $J$ decreases by $\le \epsilon$ (e.g., $10^{-3}$) in one step.
*   **Debugging with $\alpha$:** If the cost oscillates or spikes upwards, $\alpha$ is too large (or there is a bug). *Tip:* A sufficiently small $\alpha$ should guarantee a decrease in $J$ every step.
*   **Choosing $\alpha$:** Experiment with different magnitudes (e.g., $0.001, 0.01, 0.1, 1$). Find a value that is too small, find one that is too large, and pick the largest reasonable rate just below the diverging threshold.

---

### 7. Feature Engineering
You can design entirely new features by transforming or combining original features based on domain knowledge. This can massively improve predictive accuracy.
*   *Example:* Instead of using `frontage` ($x_1$) and `depth` ($x_2$) separately to predict house price, you can engineer a new feature `area` ($x_3 = x_1 \times x_2$).

---

### 8. Polynomial Regression
Polynomial regression allows you to fit non-linear curves to your data by raising features to different powers.

*   *Quadratic:* $f_{\vec{w},b}(x) = w_1x + w_2x^2 + b$
*   *Cubic:* $f_{\vec{w},b}(x) = w_1x + w_2x^2 + w_3x^3 + b$
*   *Square Root:* $f_{\vec{w},b}(x) = w_1x + w_2\sqrt{x} + b$

**Crucial Note on Scaling:** If $x$ ranges from 1 to 1,000, then $x^2$ ranges from 1 to 1,000,000, and $x^3$ ranges up to 1,000,000,000. Feature scaling is absolutely mandatory when implementing polynomial regression so gradient descent can function properly.
