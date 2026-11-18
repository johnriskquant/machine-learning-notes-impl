# Decision Trees & Tree Ensembles — Study Notes

## Table of Contents
- [1. Decision Tree Fundamentals](#1-decision-tree-fundamentals)
  - [1.1 Core Structure & Terminology](#11-core-structure--terminology)
  - [1.2 Inference Process](#12-inference-process)
- [2. Decision Tree Learning Process](#2-decision-tree-learning-process)
  - [2.1 Recursive Construction Overview](#21-recursive-construction-overview)
  - [2.2 Key Design Decisions & Stopping Criteria](#22-key-design-decisions--stopping-criteria)
- [3. Measuring Purity: Entropy & Information Gain](#3-measuring-purity-entropy--information-gain)
  - [3.1 Shannon Entropy & Axiomatic Derivation](#31-shannon-entropy--axiomatic-derivation)
  - [3.2 Information Gain (Reduction in Entropy)](#32-information-gain-reduction-in-entropy)
- [4. Handling Different Feature Types & Tasks](#4-handling-different-feature-types--tasks)
  - [4.1 Multi-Value Categorical Features (One-Hot Encoding)](#41-multi-value-categorical-features-one-hot-encoding)
  - [4.2 Continuous-Valued Features](#42-continuous-valued-features)
  - [4.3 Regression Trees (Variance Reduction)](#43-regression-trees-variance-reduction)
- [5. Tree Ensembles](#5-tree-ensembles)
  - [5.1 Weakness of Single Decision Trees & Ensemble Voting](#51-weakness-of-single-decision-trees--ensemble-voting)
  - [5.2 Sampling with Replacement (Bootstrapping)](#52-sampling-with-replacement-bootstrapping)
  - [5.3 Bagged Decision Trees & Random Forest](#53-bagged-decision-trees--random-forest)
  - [5.4 Boosting & XGBoost](#54-boosting--xgboost)
- [6. Decision Trees vs. Neural Networks](#6-decision-trees-vs-neural-networks)

---

## 1. Decision Tree Fundamentals

### 1.1 Core Structure & Terminology
* **Tree Representation:** A directed acyclic graph inverted from biological trees (analogous to an indoor hanging plant), where splits cascade downward from top to bottom.
* **Node Types:**
  * **Root Node:** The topmost node in the tree (depth $0$) where evaluation begins on the full dataset.
  * **Decision Nodes (Internal Nodes):** Intermediate nodes that inspect a specific input feature $x_j$ and route examples down the left or right branch based on the feature's value or threshold.
  * **Leaf Nodes (Terminal Nodes):** Bottom nodes that perform no further splits and output the final prediction ($\hat{y}$).
* **Model Selection Objective:** Out of all valid trees constructible from the training features, the learning algorithm seeks a tree that achieves high training accuracy while generalizing well to cross-validation and test sets.


---

### 1.2 Inference Process
1. Start at the **root node** with a test example $x$.
2. Evaluate the feature test specified inside the current decision node.
3. Traverse down the **left** or **right** branch matching the example's feature value.
4. Repeat steps 2–3 until reaching a **leaf node**, then output the class label (classification) or numerical value (regression) stored at that leaf.

[Back to TOC](#table-of-contents)

---

## 2. Decision Tree Learning Process

### 2.1 Recursive Construction Overview
Building a decision tree is a **top-down, greedy, recursive algorithm**:
1. Start with all $m$ training examples at the **root node**.
2. Compute the split quality (**Information Gain** for classification; **Variance Reduction** for regression) across all candidate features and thresholds.
3. Select the feature (and threshold) that maximizes purity gain, and partition the dataset into left and right subsets.
4. Recursively repeat steps 2–3 independently on the left child subset and right child subset as if each were the root of a smaller sub-tree.
5. Terminate recursion on a branch and instantiate a **leaf node** whenever a stopping condition is triggered.


---

### 2.2 Key Design Decisions & Stopping Criteria
At every node during training, the algorithm resolves two core questions: **(1) which feature to split on**, and **(2) when to stop splitting**.

* **Stopping Criteria (Pre-Pruning Hyperparameters):**
  1. **100% Node Purity:** All examples in the node belong to a single class ($H(S) = 0$).
  2. **Maximum Depth (`max_depth`):** Splitting further would exceed the allowed number of hops from the root node (root is depth $0$, children are depth $1$, etc.). Prevents overly large trees and limits overfitting.
  3. **Minimum Impurity Decrease / Gain Threshold:** The maximum achievable Information Gain (or variance reduction) across all candidate splits falls below a minimum threshold. Avoids adding complexity for negligible gains.
  4. **Minimum Samples to Split (`min_samples_split`):** The number of training examples at the node is below a specified threshold. Prevents overfitting on tiny data fragments (leaf predicts the majority class or mean value).

[Back to TOC](#table-of-contents)

---

## 3. Measuring Purity: Entropy & Information Gain

### 3.1 Shannon Entropy & Axiomatic Derivation
Imported from Claude Shannon's information theory (1948), **entropy** $H(S)$ quantifies the average uncertainty, disorder, or impurity in a set of examples $S$.

* **Three Mathematical Axioms:**
  1. **Additivity / Independence:** For independent events where probabilities multiply ($P(A \cap B) = P(A)P(B)$), total uncertainty must add; only a logarithm satisfies $\log(P(A)P(B)) = \log P(A) + \log P(B)$.
  2. **Monotonicity:** Higher-probability outcomes carry less "surprise." Using $-\log_2(p_i)$ ensures $-\log_2(1) = 0$ (zero surprise at certainty) and increases as $p_i \to 0$.
  3. **Maximization:** Uncertainty peaks when all classes are equally probable (uniform distribution).

* **Step-by-Step Formula:**
  * **Surprise of class $i$:** $\text{Surprise}(i) = -\log_2(p_i)$
  * **Multi-Class Entropy ($C$ classes):**
    $$H(S) = -\sum_{i=1}^{C} p_i \log_2(p_i)$$
  * **Binary Classification Entropy ($p_1 = p$, $p_0 = 1 - p$):**
    $$H(p_1) = -p_1 \log_2(p_1) - (1 - p_1) \log_2(1 - p_1)$$

* **Key Properties:**
  * **Base-2 Logarithm:** Scales the maximum binary entropy peak to exactly $1.0$ at $p_1 = 0.5$ (50/50 split = maximum impurity). Using $\ln$ (base $e$) only rescales the vertical axis without changing split rankings.
  * **Zero-Probability Convention:** By convention, $0 \log_2(0) \equiv 0$, so completely pure nodes ($p_1 = 1$ or $p_1 = 0$) have $H = 0$.
  * *(Note: Gini impurity is an alternative impurity curve ranging from $0$ at pure nodes to $0.5$ at a 50/50 binary split that behaves very similarly in practice.)*


---

### 3.2 Information Gain (Reduction in Entropy)
Rather than comparing raw child entropies directly, **Information Gain** weights child entropies by the fraction of examples routed to each branch and subtracts the result from the parent node's entropy.

* **Notation:**
  * $p_1^{\text{root}}$: Fraction of positive examples at the parent/root node.
  * $p_1^{\text{left}}, p_1^{\text{right}}$: Fraction of positive examples in the left and right child branches.
  * $w^{\text{left}} = \frac{m_{\text{left}}}{m_{\text{root}}}$, $w^{\text{right}} = \frac{m_{\text{right}}}{m_{\text{root}}}$: Proportion of parent examples sent to the left and right branches.

* **Information Gain Formula:**
  $$\text{Information Gain} = H\left(p_1^{\text{root}}\right) - \left[ w^{\text{left}} H\left(p_1^{\text{left}}\right) + w^{\text{right}} H\left(p_1^{\text{right}}\right) \right]$$

* **Why Subtract from $H(p_1^{\text{root}})$?**
  * Minimizing weighted child entropy is mathematically identical to maximizing Information Gain for a given node, but computing the explicit **reduction** in entropy allows direct comparison against a minimum gain stopping threshold.

[Back to TOC](#table-of-contents)

---

## 4. Handling Different Feature Types & Tasks

### 4.1 Multi-Value Categorical Features (One-Hot Encoding)
* **Problem:** If a categorical feature takes on $k > 2$ discrete values (e.g., *pointy*, *floppy*, *oval* ears), splitting directly creates a $k$-way branch.
* **Solution — One-Hot Encoding:**
  * Replace a single categorical feature having $k$ possible values with $k$ binary features taking values in $\{0, 1\}$.
  * For any single training example, **exactly one** of the $k$ binary features equals $1$ (the "hot" feature) and the remaining $k - 1$ equal $0$.
  * Restores standard binary splitting in decision trees and simultaneously converts categorical features into numerical inputs usable by **neural networks**, **linear regression**, and **logistic regression**.


---

### 4.2 Continuous-Valued Features
* **Threshold Splitting:** For a numerical feature $x_j$ (e.g., animal weight), evaluate candidate binary splits of the form $x_j \le \theta$.
* **Candidate Threshold Selection:**
  1. Sort the $m$ examples at the current node by their values on feature $x_j$.
  2. Compute the $m - 1$ midpoints between adjacent sorted values as candidate thresholds $\theta$.
  3. Compute the **Information Gain** for each candidate midpoint $\theta$ and select the threshold yielding the highest gain.
  4. Compare the best continuous split's Information Gain against all other features at that node; split on the continuous feature if its gain is highest overall.


---

### 4.3 Regression Trees (Variance Reduction)
Decision trees generalize to continuous target prediction ($y \in \mathbb{R}$) by replacing class purity with numerical spread (**variance**):

| Component | Classification Tree | Regression Tree |
| :--- | :--- | :--- |
| **Leaf Prediction ($\hat{y}$)** | Majority class label in the leaf | Mean (average) of target values $y$ in the leaf |
| **Impurity Metric** | Entropy $H(S)$ (or Gini impurity) | Variance $\text{Var}(y)$ of target values in the subset |
| **Split Selection Criterion** | Maximize **Information Gain** | Maximize **Variance Reduction** |

* **Variance Reduction Formula:**
  $$\text{Variance Reduction} = \text{Var}\left(y^{\text{root}}\right) - \left[ w^{\text{left}} \text{Var}\left(y^{\text{left}}\right) + w^{\text{right}} \text{Var}\left(y^{\text{right}}\right) \right]$$

[Back to TOC](#table-of-contents)

---

## 5. Tree Ensembles

### 5.1 Weakness of Single Decision Trees & Ensemble Voting
* **High Variance / Instability:** A single decision tree is highly sensitive to small perturbations in training data. Modifying even one training example can alter the optimal root split, cascading into a completely different tree structure.
* **Tree Ensemble Solution:** Train $B$ distinct decision trees and aggregate their outputs on new test examples via **majority voting** (classification) or **averaging** (regression). Because each individual tree only casts $1$ of $B$ votes, the combined model is substantially more accurate and robust.


---

### 5.2 Sampling with Replacement (Bootstrapping)
* **Mechanism:** Given an original dataset of $m$ examples, construct a new bootstrap training set of size $m$ by drawing one example uniformly at random, recording it, and **placing it back** into the pool before drawing the next example.
* **Effect:** Each generated dataset has the same size $m$ as the original, contains duplicate instances of some examples, and omits others—creating slightly different training sets from a single source dataset.


---

### 5.3 Bagged Decision Trees & Random Forest
* **Bagged Decision Trees (Bootstrap Aggregating):**
  * For $b = 1 \dots B$: generate a bootstrap dataset of size $m$ via sampling with replacement and train an independent decision tree on it.
  * **Choosing $B$:** Typically $B \approx 100$ (commonly $64$ to $128$). Larger $B$ never degrades generalization performance, but yields diminishing accuracy gains past ~100 trees while increasing compute cost.
* **Random Forest Improvement (Feature Subsampling):**
  * **Limitation of Pure Bagging:** Even across different bootstrap samples, greedy splitting often selects the exact same dominant feature at the root and top nodes, causing trees to remain correlated.
  * **Random Forest Modification:** At **every node split**, instead of searching all $n$ features, randomly sample a subset of $k < n$ features (typically $k = \sqrt{n}$ when $n$ is large) and select the best split strictly from those $k$ candidates.
  * **Result:** De-correlates the trees across the ensemble, improving variance reduction and overall robustness.


---

### 5.4 Boosting & XGBoost
* **Intuition (Deliberate Practice):** Rather than sampling all training examples with uniform probability ($\frac{1}{m}$) independently, **boosting** trains trees sequentially and increases sampling emphasis on examples that previously built trees ($1 \dots b-1$) misclassified or predicted poorly.
* **XGBoost (Extreme Gradient Boosting):**
  * The dominant open-source boosted tree implementation in industry and competitive ML (e.g., Kaggle).
  * **Key Engineering Advantages:**
    * Assigns continuous **sample weights** directly rather than physically resampling datasets, making training substantially faster.
    * Includes **built-in regularization** in its objective function to penalize tree complexity and prevent overfitting.
    * Provides optimized default split-finding and stopping criteria.
  * **Standard Python API:** `XGBClassifier()` for classification and `XGBRegressor()` for regression (using `.fit(X, y)` and `.predict(X)`).

[Back to TOC](#table-of-contents)

---

## 6. Decision Trees vs. Neural Networks

| Dimension | Decision Trees & Tree Ensembles (e.g., XGBoost) | Neural Networks |
| :--- | :--- | :--- |
| **Best Data Modalities** | **Tabular / structured data** (spreadsheet-style categorical & continuous features) | **Unstructured data** (images, video, audio, text), **mixed data**, and competitive on **tabular data** |
| **Training Speed** | **Very fast**; accelerates rapid error-analysis and iteration loops | Slower to train, especially for deep or large-scale architectures |
| **Interpretability** | Small single trees (dozens of nodes) are human-readable; large ensembles ($100+$ trees) require post-hoc tools | Black-box representations; low direct interpretability |
| **Transfer Learning** | **Not supported**; must be trained from scratch on task data | **Well-supported**; pre-trained weights transfer effectively to small datasets |
| **Multi-Model Composition** | Trained one tree/stage at a time; harder to optimize end-to-end across chained modules | Multiple networks can be chained together and trained jointly end-to-end via **gradient descent** |
| **Practical Recommendation** | Default to **XGBoost** on tabular tasks (use a single tree only under extreme compute constraints) | Default choice for unstructured/multi-modal inputs or when transfer learning is needed |

[Back to TOC](#table-of-contents)