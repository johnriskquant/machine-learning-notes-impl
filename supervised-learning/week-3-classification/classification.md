## Week 3: Classification and Logistic Regression

### 1. Classification
In classification tasks, the output variable $y$ takes on a finite set of discrete categories rather than a continuous, infinite range of numbers.
*   **Binary Classification:** Only two possible classes (e.g., spam/ham, fraud/not fraud, malignant/benign). 
*   **Conventions:** Outputs are typically mapped to $0$ and $1$. 
    *   $0 = \text{Negative Class}$ (e.g., absent, false, not spam)
    *   $1 = \text{Positive Class}$ (e.g., present, true, spam)
    *   *Note:* "Positive/Negative" does not imply good/bad; it merely indicates the presence or absence of the subject of interest.

**Why not Linear Regression?** 
Using linear regression to fit classification data by setting a threshold (e.g., if $`\hat{y} > 0.5`$, predict $1$) performs poorly. Outliers drastically shift the line of best fit, pulling the decision boundary with it and causing incorrect classifications for obvious data points.

---

### 2. Logistic Regression
A learning algorithm specifically designed for binary classification. It restricts the output to strictly fall between $0$ and $1$.

**The Sigmoid (Logistic) Function:**
Maps any input value to a value between $0$ and $1$. 
```math
g(z) = \frac{1}{1 + e^{-z}}
```
*   If $z$ is very large and positive, $g(z) \approx 1$.
*   If $z$ is very large and negative, $g(z) \approx 0$.
*   If $z = 0$, $g(z) = 0.5$.

**Logistic Regression Formulation:**
```math
f_{\vec{w},b}(\vec{x}) = g(\vec{w} \cdot \vec{x} + b) = \frac{1}{1 + e^{-(\vec{w} \cdot \vec{x} + b)}}
```
*   **Interpretation:** The output represents the *probability* that the class $y=1$ given the input features $`\vec{x}`$, parameterized by $`\vec{w}`$ and $b$.
    *   $`f_{\vec{w},b}(\vec{x}) = P(y=1 \mid \vec{x} ; \vec{w}, b)`$

---

### 3. Decision Boundary
The decision boundary is the line (or shape) that separates where the model predicts $y=1$ from where it predicts $y=0$.

**Thresholding:**
*   Predict $`\hat{y} = 1`$ if $`f_{\vec{w},b}(\vec{x}) \ge 0.5`$. This happens when $g(z) \ge 0.5$, which means **$z \ge 0$**.
*   Predict $`\hat{y} = 0`$ if $`f_{\vec{w},b}(\vec{x}) < 0.5`$. This happens when **$z < 0$**.

Because $`z = \vec{w} \cdot \vec{x} + b`$, the exact decision boundary lies at:
```math
\vec{w} \cdot \vec{x} + b = 0
```

**Non-Linear Boundaries:**
By engineering polynomial features (e.g., $x_1^2, x_2^2$), logistic regression can learn highly complex, non-linear decision boundaries like circles, ellipses, and irregular shapes.

---

### 4. Cost Function for Logistic Regression
The squared error cost function used in linear regression does not work for logistic regression. Because the sigmoid function is non-linear, inserting it into the squared error formula creates a **non-convex** cost function with many local minima, which prevents gradient descent from finding the global minimum.

**The Logistic Loss Function:**
Instead, we define a loss function $`L(f_{\vec{w},b}(\vec{x}^{(i)}), y^{(i)})`$ for a single training example:
```math
L(f_{\vec{w},b}(\vec{x}^{(i)}), y^{(i)}) = 
\begin{cases} 
-\log(f_{\vec{w},b}(\vec{x}^{(i)})) & \text{if } y^{(i)} = 1 \\
-\log(1 - f_{\vec{w},b}(\vec{x}^{(i)})) & \text{if } y^{(i)} = 0 
\end{cases}
```

**Deriving the Cost Function (Maximum Likelihood Estimation):**
*   **Step 1: Modeling Single Probabilities (Bernoulli Trial):**
    We can write the piecewise probabilities elegantly as a single equation:
```math
    P(y \mid x) = \hat{y}^{y}(1 - \hat{y})^{(1-y)}
```
*   **Step 2: The Likelihood Function:**
    Assuming data points are Independent and Identically Distributed (I.I.D.), the total likelihood is the product of individual probabilities:
```math
    L(w) = \prod_{i=1}^{n} \hat{y}_i^{y_i}(1 - \hat{y}_i)^{(1-y_i)}
```
*   **Step 3: Taking the Logarithm (Log-Likelihood):**
    To avoid computer underflow from multiplying small probabilities, we take the natural logarithm, turning products into sums:
```math
    \log L(w) = \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
```
*   **Step 4: Minimizing the Negative Log-Likelihood:**
    Optimization algorithms minimize cost rather than maximize likelihood. We multiply by $-1$ and divide by $m$ (or $n$) to get the average error, yielding the **Binary Cross-Entropy (Log Loss)** function:

```math
J(\vec{w},b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)}\log\left(f_{\vec{w},b}(\vec{x}^{(i)})\right) + (1 - y^{(i)})\log\left(1 - f_{\vec{w},b}(\vec{x}^{(i)})\right) \right]
```

---

### 5. Gradient Descent Implementation
To minimize the logistic cost function, we apply gradient descent using the exact same structural update rules as linear regression:

Repeat until convergence:
```math
w_j = w_j - \alpha \frac{\partial}{\partial w_j} J(\vec{w},b)
```
```math
b = b - \alpha \frac{\partial}{\partial b} J(\vec{w},b)
```

The computed partial derivatives mathematically simplify to:
```math
\frac{\partial}{\partial w_j} J(\vec{w},b) = \frac{1}{m} \sum_{i=1}^{m} \left( f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)} \right) x_j^{(i)}
```
```math
\frac{\partial}{\partial b} J(\vec{w},b) = \frac{1}{m} \sum_{i=1}^{m} \left( f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)} \right)
```

**Are Linear and Logistic Regression the same?**
They *look* identical on paper, but they are completely different algorithms because the definition of $`f_{\vec{w},b}(\vec{x})`$ has changed. In linear regression, it is $`\vec{w} \cdot \vec{x} + b`$. In logistic regression, it is $`g(\vec{w} \cdot \vec{x} + b)`$ (the sigmoid function). 

---

### 6. The Problem of Overfitting and Underfitting
*   **Underfitting (High Bias):** The model fails to capture the underlying pattern of the data. It has a strong "preconception" (bias) that the data fits a simple shape (like a straight line) despite evidence to the contrary.
*   **Overfitting (High Variance):** The model tries too hard to fit every single training example perfectly. It results in a wiggly, overly complex curve. Because the model contorts to fit random noise, a slightly different training set would produce a wildly different model (high variance). It fails to **generalize** to new, unseen data.
*   **"Just Right":** The model captures the overarching pattern and generalizes well to new examples without obsessing over training noise.

### 7. Addressing Overfitting
When a model suffers from high variance, you can:
1.  **Collect more training data:** With more examples, the algorithm is forced to fit a smoother function rather than memorizing individual data points.
2.  **Feature Selection:** Use fewer features. Pick only the most relevant features and discard the rest. (Trade-off: You throw away potentially useful information).
3.  **Regularization:** Gently reduce the magnitude (impact) of the parameters $w_j$ without eliminating features entirely.

---

### 8. Regularization (L2)
Regularization discourages the learning algorithm from using overly large parameter weights by mathematically penalizing them in the cost function. Smaller weights lead to smoother, simpler curves that are less prone to overfitting.

**Regularized Cost Function:**
We add a regularization penalty term to the end of the standard cost function:
```math
J(\vec{w},b) = \text{Original Cost} + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2
```
*   **$\lambda$ (Lambda):** The Regularization Parameter. It controls the trade-off between fitting the training data perfectly and keeping the parameter weights small.
    *   If $\lambda = 0$: No regularization. The model will overfit.
    *   If $\lambda = 10^{10}$ (enormous): The weights are crushed so close to $0$ that the model becomes a flat horizontal line. The model will underfit.
*   *Note:* By convention, we divide by $2m$ so the regularization term scales with the data, making it easier to choose a consistent $\lambda$. We typically do not regularize the bias parameter $b$.

### 9. Gradient Descent with Regularization
When we apply gradient descent to the newly regularized cost function, the derivative with respect to $w_j$ gains an extra term, while the derivative for $b$ remains unchanged:

```math
\frac{\partial}{\partial w_j} J(\vec{w},b) = \left[ \frac{1}{m} \sum_{i=1}^{m} \left( f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)} \right) x_j^{(i)} \right] + \frac{\lambda}{m} w_j
```

**Intuition behind the Update Rule:**
If we rearrange the gradient descent update rule for $w_j$:
```math
w_j = w_j \left( 1 - \alpha \frac{\lambda}{m} \right) - \alpha \left( \text{original gradient} \right)
```
Because $`\alpha`$ and $`\lambda`$ are positive and $m$ is large, the term $`(1 - \alpha \frac{\lambda}{m})`$ is a fraction slightly less than $1$ (e.g., $0.9998$). 
This proves that on **every single iteration**, regularization explicitly multiplies the weight $w_j$ by $0.9998$, forcibly shrinking it bit by bit before applying the standard gradient update.
