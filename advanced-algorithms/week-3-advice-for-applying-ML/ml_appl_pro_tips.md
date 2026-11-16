## Table of Contents

* [1. Machine Learning Diagnostics, Bias/Variance, and Model Development](#1-machine-learning-diagnostics-biasvariance-and-model-development)
  * [1.1 Deciding What to Do Next (ML Diagnostics)](#11-deciding-what-to-do-next-ml-diagnostics)
  * [1.2 Evaluating a Model](#12-evaluating-a-model)
  * [1.3 Model Selection (Train / Cross-Validation / Test Split)](#13-model-selection-train--cross-validation--test-split)
  * [1.4 Diagnosing Bias and Variance](#14-diagnosing-bias-and-variance)
  * [1.5 Regularization and Bias/Variance](#15-regularization-and-biasvariance)
  * [1.6 Establishing a Baseline Level of Performance](#16-establishing-a-baseline-level-of-performance)
  * [1.7 Learning Curves](#17-learning-curves)
  * [1.8 Summary: What to Try Next](#18-summary-what-to-try-next)
  * [1.9 Bias/Variance and Neural Networks](#19-biasvariance-and-neural-networks)
* [2. Machine Learning Development Workflow, Data Engineering, and Deployment](#2-machine-learning-development-workflow-data-engineering-and-deployment)
  * [2.1 Iterative Loop of Machine Learning Development](#21-iterative-loop-of-machine-learning-development)
  * [2.2 Error Analysis](#22-error-analysis)
  * [2.3 Adding Data & Data-Centric AI](#23-adding-data--data-centric-ai)
  * [2.4 Transfer Learning](#24-transfer-learning)
  * [2.5 Full Cycle of a Machine Learning Project](#25-full-cycle-of-a-machine-learning-project)
  * [2.6 Fairness, Bias, and Ethics](#26-fairness-bias-and-ethics)
* [3. Error Metrics for Skewed Datasets](#3-error-metrics-for-skewed-datasets)
  * [3.1 Why Classification Accuracy Fails on Skewed Datasets](#31-why-classification-accuracy-fails-on-skewed-datasets)
  * [3.2 Confusion Matrix, Precision, and Recall](#32-confusion-matrix-precision-and-recall)
  * [3.3 Trading Off Precision and Recall](#33-trading-off-precision-and-recall)
  * [3.4 Combining Precision and Recall: The $F_1$ Score](#34-combining-precision-and-recall-the-f_1-score)

---

## 1. Machine Learning Diagnostics, Bias/Variance, and Model Development

### 1.1 Deciding What to Do Next (ML Diagnostics)
When a trained model makes unacceptably large errors, there are many potential changes you could try (e.g., collecting more data, changing features, tuning regularization). Investing months in the wrong direction wastes significant time.

* **Diagnostic:** A test you run to gain insight into what is or isn't working with a learning algorithm, providing actionable guidance on how to improve its performance.
* **Common Candidate Actions (e.g., Regularized Linear Regression):**
  1. Get more training examples
  2. Try a smaller set of features
  3. Get additional features
  4. Add polynomial features ($x_1^2, x_2^2, x_1 x_2$, etc.)
  5. Decrease the regularization parameter $\lambda$
  6. Increase the regularization parameter $\lambda$

---

### 1.2 Evaluating a Model
Visualizing $f(\vec{x})$ works for 1 or 2 features, but becomes impossible with higher-dimensional feature spaces ($x_1, x_2, x_3, x_4, \dots$). Instead, split the dataset systematically to measure generalization.

#### Train / Test Split
* Split data into a **Training Set** (e.g., 70%–80%, size $m_{\text{train}}$) and a **Test Set** (e.g., 20%–30%, size $m_{\text{test}}$).
* Fit parameters $\vec{w}, b$ by minimizing the regularized cost function $J(\vec{w}, b)$ on the training set only.

#### Regression Error Metrics
* **Test Error ($J_{\text{test}}$):**
  $$J_{\text{test}}(\vec{w}, b) = \frac{1}{2m_{\text{test}}} \sum_{i=1}^{m_{\text{test}}} \left(f_{\vec{w},b}(\vec{x}_{\text{test}}^{(i)}) - y_{\text{test}}^{(i)}\right)^2$$
* **Training Error ($J_{\text{train}}$):**
  $$J_{\text{train}}(\vec{w}, b) = \frac{1}{2m_{\text{train}}} \sum_{i=1}^{m_{\text{train}}} \left(f_{\vec{w},b}(\vec{x}_{\text{train}}^{(i)}) - y_{\text{train}}^{(i)}\right)^2$$
* *Note:* Neither $J_{\text{test}}$ nor $J_{\text{train}}$ includes the regularization term $\frac{\lambda}{2m}\sum w_j^2$. Low $J_{\text{train}}$ paired with high $J_{\text{test}}$ signals failure to generalize (overfitting).

#### Classification Error Metrics
* **Misclassification Error (0/1 Error):** Most commonly used in practice—measures the fraction of examples where the predicted label $\hat{y}$ does not match the ground truth $y$ (where $\hat{y} = 1$ if $f(\vec{x}) \ge 0.5$, else $0$):
  * $J_{\text{test}}$ = fraction of **test** examples misclassified.
  * $J_{\text{train}}$ = fraction of **training** examples misclassified.
* *(Alternatively, one can compute the unregularized average cross-entropy/logistic loss on the test and training sets).*

---

### 1.3 Model Selection (Train / Cross-Validation / Test Split)

#### Why Choosing a Model Using the Test Set is Flawed
If you train multiple candidate models (e.g., polynomial degrees $d = 1, 2, \dots, 10$), evaluate each on $J_{\text{test}}$, and pick the model with the lowest $J_{\text{test}}$, then $J_{\text{test}}$ becomes an **overly optimistic** (artificially low) estimate of true generalization error—because you effectively fit an extra hyperparameter ($d$) to the test set.

#### Three-Way Dataset Split
To select models fairly without contaminating the test set, split the data into three subsets:
1. **Training Set (~60%, $m_{\text{train}}$):** Used to fit model parameters $\vec{w}, b$.
2. **Cross-Validation Set / Validation Set / Dev Set (~20%, $m_{\text{cv}}$):** Used to evaluate candidate architectures/hyperparameters and select the best model.
3. **Test Set (~20%, $m_{\text{test}}$):** Held out strictly until one final model is selected; used only to report an unbiased estimate of generalization error.

#### Cross-Validation Error ($J_{\text{cv}}$)
* **Regression:**
  $$J_{\text{cv}}(\vec{w}, b) = \frac{1}{2m_{\text{cv}}} \sum_{i=1}^{m_{\text{cv}}} \left(f_{\vec{w},b}(\vec{x}_{\text{cv}}^{(i)}) - y_{\text{cv}}^{(i)}\right)^2$$
* **Classification:** Fraction of cross-validation examples misclassified (or unregularized average cross-entropy loss).

#### Model Selection Workflow (Polynomial Degree or Neural Network Architecture)
1. Train all candidate models (e.g., $d = 1 \dots 10$, or small/medium/large neural networks) on the **training set** to obtain $(\vec{w}^{(1)}, b^{(1)}), (\vec{w}^{(2)}, b^{(2)}), \dots$.
2. Evaluate every candidate model on the **cross-validation set** to compute $J_{\text{cv}}(\vec{w}^{(k)}, b^{(k)})$.
3. Select the model with the **lowest $J_{\text{cv}}$**.
4. Evaluate only that final chosen model on the **test set** ($J_{\text{test}}$) to report its generalization error.

---

### 1.4 Diagnosing Bias and Variance

Comparing $J_{\text{train}}$ and $J_{\text{cv}}$ reveals whether an algorithm is underfitting or overfitting, even when features cannot be plotted.

| Diagnosis | Training Error ($J_{\text{train}}$) | Cross-Validation Error ($J_{\text{cv}}$) | Cause |
| :--- | :--- | :--- | :--- |
| **High Bias (Underfitting)** | **High** | **High** (close to $J_{\text{train}}$) | Model is too simple to capture the underlying pattern (e.g., $d=1$ linear fit on curved data). |
| **High Variance (Overfitting)** | **Low** | **Much higher than $J_{\text{train}}$** ($J_{\text{cv}} \gg J_{\text{train}}$) | Model is overly complex and memorizes training noise (e.g., $d=4$ polynomial on a small dataset). |
| **Just Right** | **Low** | **Low** | Model complexity matches the true data structure. |
| **High Bias & High Variance** | **High** | **Much higher than $J_{\text{train}}$** ($J_{\text{cv}} \gg J_{\text{train}}$) | Can occur in neural networks: overfits part of the input space while underfitting another part. |

#### Error vs. Model Complexity ($d$)
As polynomial degree $d$ increases (without regularization):
* $J_{\text{train}}$ monotonically decreases as higher-degree curves fit training points more tightly.
* $J_{\text{cv}}$ forms a U-shaped curve: high at low $d$ (high bias), lowest at an intermediate $d$ (just right), and high again at large $d$ (high variance).

---

### 1.5 Regularization and Bias/Variance
In regularized regression:
$$J(\vec{w}, b) = \frac{1}{2m} \sum_{i=1}^{m} \left(f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)}\right)^2 + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2$$

* **Very Large $\lambda$ (e.g., $\lambda = 10,000$):** Penalizes weights heavily so $w_j \approx 0$ and $f(\vec{x}) \approx b$ (nearly flat constant line) $\implies$ **High Bias (Underfitting)** ($J_{\text{train}}$ high, $J_{\text{cv}}$ high).
* **Very Small $\lambda$ (e.g., $\lambda = 0$):** No regularization penalty $\implies$ **High Variance (Overfitting)** ($J_{\text{train}}$ low, $J_{\text{cv}} \gg J_{\text{train}}$).
* **Intermediate $\lambda$:** Balances fit and smoothness $\implies$ **Just Right** ($J_{\text{train}}$ low, $J_{\text{cv}}$ low).

#### Choosing $\lambda$ via Cross-Validation
1. Test a sequence of $\lambda$ values (e.g., $0, 0.01, 0.02, 0.04, \dots, 10$).
2. For each $\lambda$, minimize the regularized cost on the **training set** to get parameters $(\vec{w}^{(k)}, b^{(k)})$.
3. Compute the unregularized $J_{\text{cv}}(\vec{w}^{(k)}, b^{(k)})$ on the **cross-validation set** and pick the $\lambda$ with the lowest $J_{\text{cv}}$.
4. Report $J_{\text{test}}$ on the **test set** for the selected model.

*(Note: Plotting error vs. $\lambda$ produces a mirror image of error vs. polynomial degree $d$: as $\lambda$ increases, $J_{\text{train}}$ rises steadily, while $J_{\text{cv}}$ starts high at $\lambda=0$, dips at the optimal $\lambda$, and rises again at very large $\lambda$).*

---

### 1.6 Establishing a Baseline Level of Performance
To judge whether $J_{\text{train}}$ is truly "high," benchmark it against a **baseline level of performance**—the realistic error rate you can hope to achieve (which is often $> 0\%$ on noisy tasks like mobile speech recognition).

* **How to Establish a Baseline:**
  1. **Human-level performance** (ideal for unstructured data such as audio, images, or text).
  2. **Competing or prior algorithm performance**.
  3. **Domain experience / informed estimate**.

* **Measuring the Two Diagnostic Gaps:**
  1. **Bias Gap:** $J_{\text{train}} - \text{Baseline}$ (If large $\implies$ **High Bias**).
  2. **Variance Gap:** $J_{\text{cv}} - J_{\text{train}}$ (If large $\implies$ **High Variance**).

#### Speech Recognition Benchmark Examples
| Scenario | Human Baseline | $J_{\text{train}}$ | $J_{\text{cv}}$ | Bias Gap ($J_{\text{train}} - \text{Base}$) | Variance Gap ($J_{\text{cv}} - J_{\text{train}}$) | Diagnosis |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Case 1** | 10.6% | 10.8% | 14.8% | 0.2% (Small) | 4.0% (Large) | **High Variance** |
| **Case 2** | 10.6% | 15.0% | 15.5% | 4.4% (Large) | 0.5% (Small) | **High Bias** |
| **Case 3** | 10.6% | 15.0% | 19.7% | 4.4% (Large) | 4.7% (Large) | **High Bias & High Variance** |

---

### 1.7 Learning Curves
A **learning curve** plots $J_{\text{train}}$ and $J_{\text{cv}}$ against the training set size $m_{\text{train}}$.

* **General Behavior:**
  * As $m_{\text{train}}$ grows, $J_{\text{train}}$ **increases** (fitting 1–3 examples perfectly with zero error is easy; fitting hundreds of examples simultaneously without error is harder).
  * As $m_{\text{train}}$ grows, $J_{\text{cv}}$ **decreases** as the model generalizes better, remaining above $J_{\text{train}}$.

* **High Bias Learning Curve:**
  * Both $J_{\text{train}}$ and $J_{\text{cv}}$ flatten out (plateau) quickly at a high error level far above the human baseline.
  * **Key Insight:** If an algorithm suffers from **high bias**, collecting more training data by itself will **not** help much.

* **High Variance Learning Curve:**
  * $J_{\text{train}}$ stays very low (potentially even below human-level error due to overfitting), while $J_{\text{cv}}$ remains much higher with a wide gap between them.
  * Extrapolating to larger $m_{\text{train}}$ shows $J_{\text{cv}}$ continuing to drop toward $J_{\text{train}}$.
  * **Key Insight:** If an algorithm suffers from **high variance**, getting more training data **is likely to help** significantly.

*(Note: Explicitly plotting learning curves across many subset sizes is computationally expensive, so practitioners mostly use this curve behavior as a mental framework alongside single-point $J_{\text{train}}$ and $J_{\text{cv}}$ evaluations).*

---

### 1.8 Summary: What to Try Next

* **Fixes for High Variance (Overfitting — simplify model or add data):**
  1. Get more training examples
  2. Try a smaller set of features (remove irrelevant or redundant features)
  3. Increase $\lambda$ (increase regularization)

* **Fixes for High Bias (Underfitting — increase model capacity/features):**
  1. Get additional features
  2. Add polynomial features ($x_1^2, x_2^2, x_1 x_2$, etc.)
  3. Decrease $\lambda$ (reduce regularization)

*(Warning: Never randomly discard training data to fix high bias—it only worsens cross-validation and generalization error).*

---

### 1.9 Bias/Variance and Neural Networks
Classical models face a **bias-variance trade-off** where reducing bias increases variance and vice versa. Deep learning changes this dynamic because **large neural networks trained on small-to-moderate datasets act as low-bias machines**—when properly regularized, increasing network size rarely hurts performance.

#### Iterative Neural Network Development Loop
1. Train the network and check: **Does it do well on the training set?** ($J_{\text{train}} \approx \text{Baseline}$?)
   * **No (High Bias):** Use a **bigger neural network** (more hidden layers or more units per layer) and retrain.
2. Once $J_{\text{train}}$ is low, check: **Does it do well on the cross-validation set?** ($J_{\text{cv}} \approx J_{\text{train}}$?)
   * **No (High Variance):** Get **more training data** (or add regularization) and loop back to Step 1.
   * **Yes:** Done! Evaluate on the test set.
* **Caveats:** Very large networks increase training and inference compute costs (requiring GPUs), and collecting additional data eventually hits practical limits.

#### Regularization in Neural Networks
Adding L2 regularization to a large neural network allows it to perform **as well as or better than** a smaller network (the only trade-off is slower computation):
$$J(W, B) = \frac{1}{m} \sum_{i=1}^{m} L\left(f_{W,B}(\vec{x}^{(i)}), y^{(i)}\right) + \frac{\lambda}{2m} \sum_{\text{all weights}} w^2$$
*(Regularization is applied to all weights $W$ across layers; bias parameters $b$ are typically left unregularized).*

**TensorFlow Implementation:**
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import L2

model = Sequential([
    Dense(units=25, activation='relu', kernel_regularizer=L2(0.01)),
    Dense(units=15, activation='relu', kernel_regularizer=L2(0.01)),
    Dense(units=1, activation='sigmoid', kernel_regularizer=L2(0.01))
])
```

[↑ Back to Table of Contents](#table-of-contents)

---

## 2. Machine Learning Development Workflow, Data Engineering, and Deployment

### 2.1 Iterative Loop of Machine Learning Development
Building an effective machine learning system is an iterative process rather than a single linear pass. A newly trained model rarely works well on the first attempt.

1. **Choose Overall System Architecture:** Select the model type, initial dataset, features, and hyperparameters.
2. **Implement and Train Model:** Fit the model parameters on the training set.
3. **Run Diagnostics:** Evaluate bias/variance ($J_{\text{train}}$ vs. $J_{\text{cv}}$) and perform **error analysis** to identify where and why the model fails.
4. **Iterate:** Use diagnostic insights to update the architecture, hyperparameters, features, or dataset, then repeat the loop until target performance is achieved.

#### Case Study: Email Spam Classifier (Text Classification)
* **Problem Setup:** Classify an email ($x$) as spam ($y=1$) or non-spam ($y=0$).
* **Feature Construction:** Build a vocabulary list of the top $10,000$ words (e.g., *a, andrew, buy, deal, discount, ...*) and represent each email as a $10,000$-dimensional feature vector $\vec{x}$:
  * *Binary encoding:* $x_j \in \{0, 1\}$ indicating whether word $j$ appears in the email.
  * *Frequency encoding:* $x_j$ equals the count of how many times word $j$ appears.
* **Candidate Improvement Ideas:**
  * Collect more spam emails (e.g., via "honeypot" fake email addresses exposed to spammers).
  * Engineer features from email routing headers (the sequence of servers/networks the email traversed).
  * Improve text features via word stemming/grouping (e.g., treating *discount* and *discounting* as the same root word).
  * Build detectors for deliberate misspellings (e.g., *w4tches*, *med1cine*, *m0rtgage*).
* **Role of Diagnostics:** Diagnostics prevent wasting months on low-impact ideas (e.g., collecting more data when the model suffers from high bias).

---

### 2.2 Error Analysis
**Error analysis** is the process of manually examining misclassified examples from the cross-validation set to uncover patterns and prioritize the highest-impact improvements.

#### Workflow
1. Gather misclassified examples from the cross-validation set (if there are too many—e.g., $1,000$ errors out of $5,000$ examples—take a random sample of ~100 to 200 misclassified examples).
2. Read through each error manually and group them into common themes or traits.
3. Count how many errors fall into each category. *(Note: Categories are not mutually exclusive; a single email can belong to multiple categories).*

#### Example Error Analysis Table (100 Misclassified Spam Emails)
| Error Category | Count (out of 100) | Priority & Actionable Insight |
| :--- | :--- | :--- |
| **Pharmaceutical spam** (selling drugs/medicines) | 21 | **High Priority:** Collect more pharma spam examples; add features for pharmaceutical/drug names. |
| **Phishing** (stealing passwords) | 18 | **High Priority:** Collect more phishing examples; write code to extract features from suspicious URLs. |
| **Unusual email routing** | 7 | **Moderate Priority:** Inspect email header paths for anomalous server hops. |
| **Deliberate misspellings** | 3 | **Low Priority:** Even a perfect misspelling detector eliminates at most 3% of errors. |
| **Embedded image spam** (text inside images) | — | Inspect image attachments rather than relying solely on body text. |

* **Limitation:** Error analysis works best on problems where humans can easily determine the correct label and understand why an example is hard (e.g., spam, images, audio). It is much harder on tasks where humans lack intuition (e.g., predicting which website ad a user will click).

---

### 2.3 Adding Data & Data-Centric AI

#### Targeted Data Collection
* Instead of blindly collecting more data of all types (which is slow and expensive), focus on acquiring more examples specifically for the underperforming categories identified in error analysis (e.g., having labelers scan unlabeled emails specifically for pharmaceutical or phishing spam).

#### Data Augmentation
**Data augmentation** creates new training examples by applying realistic transformations or distortions to existing $(x, y)$ pairs without changing the ground-truth label $y$.
* **Images (e.g., Optical Character Recognition):**
  * Rotate, enlarge, shrink, adjust contrast, apply mirror reflections (for symmetric letters), or apply random grid warpings.
* **Audio (e.g., Speech Recognition):**
  * Superimpose clean speech clips onto representative background noise (crowd chatter, car noise) or simulate low-quality cellular connections.
* **Guiding Principle:** The synthetic distortions must reflect the types of noise and variation expected in the **test set**. Adding purely random, unrepresentative noise (e.g., random per-pixel salt-and-pepper noise on images) rarely helps.

#### Data Synthesis
**Data synthesis** generates brand-new training examples from scratch rather than modifying existing ones.
* *Example (Photo OCR):* Rendering random text strings on a computer using hundreds of different fonts, colors, and contrast levels to generate realistic character images at scale.
* Most commonly used in computer vision tasks.

#### Model-Centric vs. Data-Centric AI
* **Model-Centric Approach:** Holding the dataset fixed while iterating primarily on model architecture and algorithm code.
* **Data-Centric Approach:** Leveraging established, high-performing model architectures while focusing engineering effort on systematically improving data quality, targeted coverage, augmentation, and synthesis.

---

### 2.4 Transfer Learning
**Transfer learning** leverages parameters learned from a large dataset on a related task to achieve strong performance on a target application with very little training data.

#### Two-Step Process
1. **Supervised Pre-Training:** Train a neural network on a massive dataset (e.g., $1,000,000$ images across $1,000$ classes such as cats, dogs, cars, and people) to learn layer parameters $(W^{[1]}, b^{[1]}), \dots, (W^{[L]}, b^{[L]})$—or download a freely available open-source pre-trained model.
2. **Fine-Tuning:** Copy the pre-trained hidden layers, replace the original output layer with a new output layer matching your target task (e.g., $10$ units for digits $0\text{--}9$) with newly initialized weights, and train on your smaller dataset:
   * **Option 1 (Train Output Layer Only):** Freeze all pre-trained hidden layer parameters and update *only* the new output layer weights. Best when your target dataset is **very small**.
   * **Option 2 (Fine-Tune All Layers):** Initialize hidden layers with the pre-trained weights and update *all* parameters across the entire network using gradient descent or Adam. Best when your target dataset is **somewhat larger**.

#### Why Transfer Learning Works
* Neural networks learn hierarchical features:
  * **Layer 1:** Detects low-level features (edges).
  * **Layer 2:** Combines edges to detect corners.
  * **Layer 3+:** Combines corners to detect curves and basic shapes.
* These generic visual/structural representations transfer directly to new tasks within the same domain.
* **Key Constraint:** The input modality/type $x$ must be the same for pre-training and fine-tuning (e.g., image-to-image, audio-to-audio, or text-to-text).

---

### 2.5 Full Cycle of a Machine Learning Project
Training a model is only one stage of building a production machine learning system:

1. **Scope Project:** Define the business/user problem, success metrics, inputs $x$, and outputs $y$.
2. **Collect Data:** Identify required data sources and gather labeled training, cross-validation, and test sets.
3. **Train Model (Iterative Loop):** Train the model, run bias/variance diagnostics and error analysis, and perform targeted data collection/augmentation until accuracy meets deployment standards.
4. **Deploy in Production:**
   * Host the trained model on an **inference server**.
   * Client applications (e.g., mobile apps) send input $x$ via an **API call** to the inference server, which runs the model and returns prediction $\hat{y}$.
   * Requires software engineering for latency optimization, compute cost efficiency, reliability, and scaling.
5. **Monitor & Maintain System (MLOps):**
   * Log incoming inputs $x$ and predictions $\hat{y}$ (subject to user privacy and consent).
   * Monitor for **data drift** (e.g., new celebrity names or newly elected politicians appearing in voice searches that were absent from historical training data) and trigger retraining/model updates when accuracy degrades.
   * **MLOps (Machine Learning Operations):** The engineering discipline dedicated to systematically building, deploying, scaling, monitoring, and maintaining ML models in production.

---

### 2.6 Fairness, Bias, and Ethics
Machine learning systems impact billions of people and can cause significant harm if deployed without ethical safeguards.

#### Documented Risks and Adverse Use Cases
* **Algorithmic Discrimination & Stereotyping:**
  * Automated hiring tools discriminating against women.
  * Facial recognition models disproportionately misidentifying dark-skinned individuals.
  * Biased credit/loan approval systems disadvantaging demographic subgroups.
  * Search algorithms reinforcing negative gender or racial stereotypes.
* **Malicious / Adverse Applications:**
  * Non-consensual deepfake videos and synthetic media without disclosure.
  * Social media engagement optimizers amplifying toxic or incendiary speech.
  * Automated bots generating fake product reviews or political disinformation.
  * ML-driven financial fraud and spam attacks.

#### Guidelines for Ethical and Fair ML Systems
1. **Assemble Diverse Teams:** Bring together individuals across diverse genders, ethnicities, cultures, and backgrounds to brainstorm potential failure modes and harms to vulnerable groups before deployment.
2. **Consult Industry Standards:** Research established fairness frameworks, compliance rules, and sector-specific guidelines (e.g., fair lending standards in financial services).
3. **Conduct Pre-Deployment Audits:** Evaluate model performance across protected demographic subgroups prior to production release to uncover and remediate hidden biases.
4. **Establish Mitigation & Rollback Plans:** Prepare immediate fallback mechanisms (such as reverting to a known fair prior system) and continuously monitor live deployments for emerging harms.
5. **Uphold Ethical Boundaries:** Refuse or terminate projects that make society worse off, even if the financial case is sound.

[↑ Back to Table of Contents](#table-of-contents)

---

## 3. Error Metrics for Skewed Datasets

### 3.1 Why Classification Accuracy Fails on Skewed Datasets
When the ratio of positive ($y = 1$, the rare class) to negative ($y = 0$) examples is heavily skewed (far from 50–50), standard error metrics like overall classification accuracy become misleading.

* **Example (Rare Disease Detection):**
  * Suppose only **$0.5\%$** of patients in a population actually have a rare disease ($y = 1$).
  * A trained ML model achieves **$1\%$ test error** ($99\%$ accuracy).
  * A trivial, non-learning baseline that simply prints `y = 0` for every patient achieves **$0.5\%$ test error** ($99.5\%$ accuracy).
* **Problem:** Comparing raw accuracy ($99.5\%$ vs. $99.2\%$ vs. $99.6\%$) cannot distinguish a genuinely useful diagnostic model from a degenerate model that never predicts $y = 1$.

---

### 3.2 Confusion Matrix, Precision, and Recall
To evaluate models on skewed classes, organize predictions on the cross-validation or test set into a $2 \times 2$ **Confusion Matrix** (with $y = 1$ assigned to the rare class):

| | **Actual Class: $1$** | **Actual Class: $0$** |
| :--- | :--- | :--- |
| **Predicted Class: $1$** | **True Positive (TP)** *(e.g., 15)* | **False Positive (FP)** *(e.g., 5)* |
| **Predicted Class: $0$** | **False Negative (FN)** *(e.g., 10)* | **True Negative (TN)** *(e.g., 70)* |

#### Precision
Of all examples where the model **predicted** $y = 1$, what fraction actually belongs to the positive class?
$$\text{Precision} = \frac{\text{True Positives}}{\text{Total Predicted Positives}} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
* *Example:* $\frac{15}{15 + 5} = \frac{15}{20} = 0.75$ ($75\%$).

#### Recall
Of all examples that **actually** belong to the positive class ($y = 1$), what fraction did the model correctly detect?
$$\text{Recall} = \frac{\text{True Positives}}{\text{Total Actual Positives}} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
* *Example:* $\frac{15}{15 + 10} = \frac{15}{25} = 0.60$ ($60\%$).

#### Spotting Degenerate Models
* If an algorithm always predicts $y = 0$, then $\text{TP} = 0$, yielding $\text{Recall} = 0$ (and $\text{Precision} = \frac{0}{0}$, which is defined as $0$ in practice).
* Requiring **both** precision and recall to be reasonably high ensures the classifier makes reliable positive predictions while catching a meaningful share of actual positive cases.

---

### 3.3 Trading Off Precision and Recall
In binary classification (e.g., logistic regression outputting $f(\vec{x}) \in [0, 1]$), predictions are made by comparing $f(\vec{x})$ against a decision threshold (default $= 0.5$):
$$\hat{y} = \begin{cases} 1 & \text{if } f(\vec{x}) \ge \text{threshold} \\ 0 & \text{if } f(\vec{x}) < \text{threshold} \end{cases}$$

Adjusting this threshold directly trades off precision against recall:

| Threshold Adjustment | Example Use Case | Effect on Precision | Effect on Recall |
| :--- | :--- | :--- | :--- |
| **Raise Threshold** *(e.g., $0.7$ or $0.9$)* | Treatment is invasive or expensive; predict $y=1$ only when highly confident. | **Increases** *(fewer false positives)* | **Decreases** *(misses more true cases)* |
| **Lower Threshold** *(e.g., $0.3$)* | Treatment is safe/cheap, but untreated disease is dangerous; predict $y=1$ when in doubt. | **Decreases** *(more false positives)* | **Increases** *(catches more true cases)* |

* **Precision-Recall Curve:** Plotting $(P, R)$ across thresholds from $0.01$ to $0.99$ visualizes the trade-off curve. Selecting the operating threshold is a domain/business decision balancing false positive vs. false negative costs (rather than something tuned automatically via cross-validation).

---

### 3.4 Combining Precision and Recall: The $F_1$ Score
When comparing multiple candidate models, having two separate metrics ($P$ and $R$) can make model selection ambiguous if neither model dominates on both metrics.

#### Why Simple Arithmetic Mean Fails
Taking $\frac{P + R}{2}$ can rank a useless classifier highest—for example, a model that always predicts $y = 1$ achieves $\text{Recall} = 1.0$ with near-zero precision, producing an artificially high average around $0.51$.

#### $F_1$ Score (Harmonic Mean)
The **$F_1$ Score** combines precision and recall while placing much heavier weight on whichever metric is **lower**, penalizing models where either $P$ or $R$ is near zero:
$$F_1 = \frac{1}{\frac{1}{2}\left(\frac{1}{P} + \frac{1}{R}\right)} = \frac{2PR}{P + R}$$

#### Model Comparison Example
| Model | Precision ($P$) | Recall ($R$) | Simple Average $\frac{P+R}{2}$ | **$F_1$ Score** $\frac{2PR}{P+R}$ | Assessment |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Algorithm 1** | $0.50$ | $0.40$ | $0.45$ | **$0.444$** | **Best overall balance** |
| **Algorithm 2** | $0.70$ | $0.10$ | $0.40$ | **$0.175$** | Penalized for very low recall |
| **Algorithm 3** | $0.02$ | $1.00$ | $0.51$ | **$0.0392$** | Degenerate (predicts almost all $1$s) |

---

[↑ Back to Table of Contents](#table-of-contents)