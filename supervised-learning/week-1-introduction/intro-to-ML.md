## Week 1: Introduction to Machine Learning

### 1. What is Machine Learning?
The field of study that gives computers the ability to learn without being explicitly programmed.

### 2. Supervised Learning
Algorithms that learn an $x$ (input) $\to$ $y$ (output label) mapping. It learns from given examples that have the "right answers" (the label $y$).
*   **Regression:** Predicts a continuous number (from infinitely many possible numbers).
*   **Classification:** Predicts discrete categories (e.g., breast cancer detection: 0 = benign, 1 = malignant). It predicts a finite/small limited set of possible outputs. Categories can be numeric or non-numeric.

### 3. Unsupervised Learning
Finds some structure, pattern, or something interesting in unlabeled data. We are not giving the algorithm the "right answer" to the input. Data only comes with inputs $x$, but not output labels $y$.
*   **Clustering:** Group similar data points together (e.g., Google News grouping related articles, DNA microarray analysis, market segmentation).
*   **Anomaly Detection:** Find unusual data points (e.g., fraud detection).
*   **Dimensionality Reduction:** Compress data using fewer numbers.

---

### 4. Linear Regression Model
Fitting a straight line to data.

**Notation:**
*   **Training set:** Data used to train the model.
*   $x$ = "input" variable / feature
*   $y$ = "output" variable / "target" variable
*   $m$ = number of training examples
*   $(x, y)$ = single training example
*   $(x^{(i)}, y^{(i)})$ = $i$-th training example (1st, 2nd, 3rd...). $i$ is the index of the training set (refers to row $i$ of the data table).

**The Process:**
Training set (features, targets) $\to$ Learning algorithm $\to$ $f$ (model/function) $\to$ $\hat{y}$ (prediction).
We represent the model as $f_{w,b}(x) = wx + b$. We use the linear function as a foundation for other complex functions.

---

### 5. Cost Function
In order to implement linear regression, the first key step is to define a cost function. 

Given a training set $(X,Y)$, the algorithm should choose values for parameters (weights) $w$ and (bias) $b$ such that the linear function fits the data well (passing through the training set). The model makes a prediction: $\hat{y}^{(i)} = f_{w,b}(x^{(i)})$.

**Squared Error Cost Function:**
We need to measure how well a line fits the training set. 

$$
J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right)^2
$$

Rewritten with our model function:

$$
J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)^2
$$

**Cost Function Intuition:** 
*   We want to minimize $J(w,b)$ as a function of $w,b$.
*   If we set $b=0$, we can visualize the function $J(w)$ for each $w$ used in our model to fit the data.

### 6. Gradient Descent
An algorithm used to minimize any cost function $J(w,b)$.

**Outline:**
1. Start with some initial $w, b$ (e.g., set $w = 0, b = 0$).
2. Keep changing $w, b$ to reduce $J(w,b)$ until we settle at or near a minimum (some functions may have $>1$ minimum).

**Simultaneous Update Rule:**
Repeat until the algorithm converges (when $w, b$ don't change much with additional steps):

```python
# Python implementation of simultaneous update
tmp_w = w - alpha * dj_dw
tmp_b = b - alpha * dj_db

w = tmp_w
b = tmp_b
```

*   $\alpha$ = **Learning Rate** $\in [0,1]$. 
*   The derivative determines the direction of the step (positive or negative slope), and combined with $\alpha$, determines the size of the step.

**Learning Rate ($\alpha$) Impacts:**
*   **Too small:** Gradient descent may be too slow.
*   **Too large:** May overshoot, never reach the minimum, and fail to converge or diverge.
*   *Note:* You can reach a local minimum with a fixed learning rate because as you approach the minimum, the derivative approaches 0, meaning gradient descent automatically takes smaller steps.

**Gradient Descent for Linear Regression:**
Has a single global minimum (no local minima) because the squared error cost function is a convex function (shaped like a hammock/bowl).

Partial derivatives for the updates:

$$
\frac{\partial}{\partial w} J(w,b) = \frac{1}{m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right) x^{(i)}
$$

$$
\frac{\partial}{\partial b} J(w,b) = \frac{1}{m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)
$$

**Batch Gradient Descent:** 
"Batch" means that each step of gradient descent uses *all* the training examples ($m$) rather than just a subset of the training data.
