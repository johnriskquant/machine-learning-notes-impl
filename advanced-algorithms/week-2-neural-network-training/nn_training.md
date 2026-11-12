## Week 2: Training a Neural Network

### 1. The Three Steps of Training 
Training a neural network follows the exact same three foundational steps used to train a logistic regression model, but scaled up for multiple layers and neurons.

**Recall: Logistic Regression Steps**
1.  **Specify the model:** Define how to compute the output given input $x$ and parameters $w, b$ (e.g., $`f_{\vec{w},b}(\vec{x}) = g(\vec{w} \cdot \vec{x} + b)`$).
2.  **Specify the loss and cost functions:** Define the single-example Loss $L(f(\vec{x}), y)$ and the average Cost $J(\vec{w},b)$ over the entire dataset.
3.  **Minimize the cost function:** Use an optimization algorithm like Gradient Descent to iteratively update the parameters and minimize $J$.

**Neural Network Equivalent in TensorFlow:**
1.  **Specify the architecture:** Tell TensorFlow how to compute the inferences by sequentially linking layers.
2.  **Compile the model:** Specify the exact loss function to calculate the error.
3.  **Fit the model:** Let TensorFlow automatically run the optimization algorithm (Gradient Descent via Backpropagation) to minimize the cost.

---

### 2. Step 1: Model Specification (Architecture)
This step defines how the neural network computes its forward propagation. It tells TensorFlow exactly how many layers, neurons, and activation functions to use.

*Example: Handwritten digit recognition (binary classification: 0 or 1)*
*   Input: $x$ (image pixel array)
*   Hidden Layer 1: 25 units, Sigmoid activation
*   Hidden Layer 2: 15 units, Sigmoid activation
*   Output Layer: 1 unit, Sigmoid activation

**TensorFlow Code:**
```python
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense

    model = Sequential([
        Dense(units=25, activation='sigmoid'),
        Dense(units=15, activation='sigmoid'),
        Dense(units=1, activation='sigmoid')
    ])
```
*Note: This structure automatically defines all the weight matrices ($`W^{[1]}, W^{[2]}, W^{[3]}`$) and bias vectors ($`B^{[1]}, B^{[2]}, B^{[3]}`$).*

---

### 3. Step 2: Compiling the Model (Loss & Cost Functions)
Once the architecture is defined, you must tell the network how to measure its mistakes.

**The Loss Function ($L$):** Measures the error on a *single* training example.
*   **For Binary Classification:** Use **Binary Cross-Entropy Loss** (mathematically identical to the logistic loss).
    $$L(f(\vec{x}), y) = -y \log(f(\vec{x})) - (1-y) \log(1-f(\vec{x}))$$
    *TensorFlow syntax:* `tf.keras.losses.BinaryCrossentropy()`
*   **For Regression:** Use **Mean Squared Error (MSE) Loss**.
    $$L(f(\vec{x}), y) = \frac{1}{2}(f(\vec{x}) - y)^2$$
    *TensorFlow syntax:* `tf.keras.losses.MeanSquaredError()`

**The Cost Function ($J$):** The average of the loss over all $m$ training examples. It is a function of all weights ($W$) and biases ($B$) in the entire network.
$$J(W, B) = \frac{1}{m} \sum_{i=1}^{m} L(f_{W,B}(\vec{x}^{(i)}), y^{(i)})$$

**TensorFlow Code:**
```python
    # Compiling the model for a binary classification task
    model.compile(loss=tf.keras.losses.BinaryCrossentropy())
```
---

### 4. Step 3: Fitting the Model (Optimization)
The final step is to minimize the cost function $J(W, B)$ with respect to all parameters in the network.

**Gradient Descent in Neural Networks:**
For every layer $l$ and every unit $j$, the parameters are updated repeatedly:
$$W_{l,j} = W_{l,j} - \alpha \frac{\partial}{\partial W_{l,j}} J(W,B)$$
$$b_{l,j} = b_{l,j} - \alpha \frac{\partial}{\partial b_{l,j}} J(W,B)$$

**Backpropagation:**
To run Gradient Descent, you need the partial derivatives of the cost function with respect to every single weight and bias. TensorFlow automatically calculates these complex derivatives using a highly efficient calculus algorithm called **Backpropagation**.

**Epochs:**
An epoch is a technical term for one complete pass of the learning algorithm over the entire training dataset.

**TensorFlow Code:**
```python
    # Train the model for 100 iterations (epochs)
    model.fit(X, Y, epochs=100)
```
---

## Activation Functions in Neural Networks

### 1. Types of Activation Functions
Initially, neural networks relied heavily on the **Sigmoid** activation function because they were built by stringing together logistic regression units. However, modern neural networks use a variety of activation functions to handle different types of data and improve learning efficiency.

*   **Sigmoid Function:** 
    *   $g(z) = \frac{1}{1 + e^{-z}}$
    *   Outputs values between 0 and 1.
*   **ReLU (Rectified Linear Unit):** 
    *   $g(z) = \max(0, z)$
    *   Outputs 0 if $z$ is negative; outputs $z$ (a straight 45-degree line) if $z$ is zero or positive.
    *   Ideal for modeling non-negative continuous variables (e.g., degree of product awareness, house prices).
*   **Linear Activation Function:** 
    *   $g(z) = z$
    *   Sometimes referred to as using "no activation function".

---

### 2. Choosing Activation Functions

#### For the Output Layer
The choice for the output layer is typically dictated by the ground truth label ($y$) of your specific problem:
*   **Binary Classification ($y \in \{0, 1\}$):** Use **Sigmoid**. (Outputs the probability that $y=1$).
*   **Regression - Any Value ($y \in \mathbb{R}$):** Use **Linear**. (e.g., predicting stock price changes, which can go up or down).
*   **Regression - Non-Negative Values ($y \ge 0$):** Use **ReLU**. (e.g., predicting house prices, which cannot be negative).

#### For the Hidden Layers
**ReLU** is overwhelmingly the most common default choice for hidden layers today. 
*   **Why ReLU over Sigmoid?**
    1.  *Computational Efficiency:* ReLU is much faster to compute ($\max(0, z)$) compared to Sigmoid (which requires exponentiation).
    2.  *Faster Learning (Gradient Descent):* The Sigmoid function goes flat at both extremes (very high or very low values), which causes the gradient to vanish and slows down Gradient Descent. ReLU only goes flat on the left side (negative values), which generally allows the neural network to learn much faster.
*   *(Note: Other functions like Leaky ReLU, Tanh, or Swish are sometimes used in research, but standard ReLU is generally sufficient for most applications).*

**TensorFlow Implementation:**
```python
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense

    model = Sequential([
        Dense(units=25, activation='relu'),    # Hidden layer: ReLU
        Dense(units=15, activation='relu'),    # Hidden layer: ReLU
        Dense(units=1, activation='sigmoid')   # Output layer: Sigmoid (for binary classification)
    ])
```
---

### 3. Why We Need Non-Linear Activation Functions
What happens if we just use a **Linear** activation function ($g(z) = z$) for all hidden layers? 

The entire neural network collapses into a simple linear regression model, defeating the purpose of having multiple layers. 

**Mathematical Proof (1 hidden layer example):**
1.  **Hidden Layer:** $a^{[1]} = w^{[1]}x + b^{[1]}$ 
2.  **Output Layer:** $a^{[2]} = w^{[2]}a^{[1]} + b^{[2]}$
3.  **Substitution:** 
    $a^{[2]} = w^{[2]}(w^{[1]}x + b^{[1]}) + b^{[2]}$
    $a^{[2]} = (w^{[2]}w^{[1]})x + (w^{[2]}b^{[1]} + b^{[2]})$
4.  **Conclusion:** Let $W = w^{[2]}w^{[1]}$ and $B = w^{[2]}b^{[1]} + b^{[2]}$. The output simplifies to $a^{[2]} = Wx + B$. 

*   A linear function of a linear function is just another linear function.
*   If you use linear activations in hidden layers and a linear output, the network is equivalent to **Linear Regression**.
*   If you use linear activations in hidden layers and a Sigmoid output, the network is equivalent to standard **Logistic Regression**.

**Golden Rule:** Never use the linear activation function in hidden layers. Default to ReLU to allow the network to learn complex, non-linear features.

---

## Advanced Classification: Multi-class, Softmax, and Multi-label

### 1. Multi-class Classification
Multi-class classification applies when the target variable $y$ can take on a small number of discrete categories, but **more than two** (e.g., $y \in \{1, 2, 3, 4\}$).
*   **Examples:** Recognizing handwritten digits (0-9), diagnosing 5 different diseases, or classifying manufacturing defects (scratch, discoloration, chip).
*   **Goal:** Instead of just finding the probability of one class, the model estimates the probability of the input belonging to *each* possible class: $P(y=1|x), P(y=2|x), \dots, P(y=N|x)$.
*   **Decision Boundary:** The learning algorithm learns complex decision boundaries that divide the feature space into multiple distinct regions rather than just two.

---

### 2. Softmax Regression
Softmax regression is a direct generalization of logistic regression to handle multi-class classification.

**Formulation for $N$ classes:**
1.  Compute the linear combinations (logits) for each class $j$:
    $$z_j = \vec{w}_j \cdot \vec{x} + b_j$$
2.  Apply the Softmax function to compute the probability activation $a_j$ for each class:
    $$a_j = \frac{e^{z_j}}{\sum_{k=1}^{N} e^{z_k}}$$

*   **Interpretation:** $a_j$ is the estimated probability that $y=j$ given the input $x$.
*   **Property:** By definition of the formula, all activations $a_1 + a_2 + \dots + a_N$ will always sum up exactly to $1$. 

**Cost Function (Cross-Entropy Loss):**
For a single training example where the true ground truth label is $y$, the loss is:
$$L = -\log(a_y)$$

*   If the true class is $j$, the loss is simply $-\log(a_j)$. 
*   **Intuition:** If the model predicts a high probability for the correct class (e.g., $a_j \approx 1$), the loss is near 0. If it predicts a low probability, the loss approaches infinity.

---

### 3. Neural Networks with Softmax
To implement multiclass classification in a neural network, place the Softmax function in the **output layer**.

*   **Architecture (e.g., Digit Recognition 0-9):**
    *   Input layer: $\vec{x}$
    *   Hidden layer 1: 25 units, ReLU
    *   Hidden layer 2: 15 units, ReLU
    *   Output layer: **10 units, Softmax** (outputs $a_1, \dots, a_{10}$)
*   **A Unique Property of Softmax:** Unlike Sigmoid or ReLU, where the activation $a_j$ only depends on its corresponding $z_j$, a Softmax activation $a_j$ depends on **all** $z$ values in that layer simultaneously (because the denominator sums over all $z_k$).

**TensorFlow Target Formats & Loss Functions:**
*   **`SparseCategoricalCrossentropy`:** Use this when your target $y$ is an integer index (e.g., $y=2$). "Sparse" implies each example belongs to exactly one category.
*   **`CategoricalCrossentropy`:** Use this when your target $y$ is one-hot encoded (e.g., $y = [0, 0, 1, 0, \dots, 0]$).

---

### 4. Improved Implementation (Numerical Stability)
Computers have finite memory to store floating-point numbers. Explicitly computing the softmax probabilities ($a$) and then plugging them into the log loss function ($-\log(a)$) can cause severe **numerical round-off errors** (overflow/underflow) if $z$ values are very large or very small.

**The Preferred "Logits" Method:**
Instead of calculating $a$ as an intermediate step, we can pass the raw $z$ values (called **logits**) directly into the loss function. TensorFlow can internally rearrange the mathematical terms to compute the loss with much higher numerical stability.

**Recommended TensorFlow Code:**
```python
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense

    # 1. Build the model with a LINEAR output layer (no softmax)
    model = Sequential([
        Dense(units=25, activation='relu'),
        Dense(units=15, activation='relu'),
        Dense(units=10, activation='linear') # Output is raw z (logits)
    ])

    # 2. Compile with from_logits=True
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    )

    model.fit(X, Y, epochs=100)
```

**Handling Predictions:**
Because the output layer is now linear, the model outputs raw logits (from negative to positive infinity), not probabilities. 
*   If you just need the final prediction class, you can find the index of the largest output using `np.argmax(logits)`.
*   If you explicitly need probability percentages, you must manually pass the output logits through a softmax function: `tf.nn.softmax(logits)`.

---

### 5. Multi-label Classification
Multi-label classification is fundamentally different from multi-class classification. 
*   **Multi-class:** An input belongs to exactly *one* of many categories (e.g., is this image a 1, a 2, OR a 3?).
*   **Multi-label:** An input can belong to *multiple* categories simultaneously (e.g., does this image contain a car AND a pedestrian AND a bus?).

**Target Vector:**
The target $Y$ is a vector of binary numbers for a single input, e.g., $Y = [1, 0, 1]$ (Car=Yes, Bus=No, Pedestrian=Yes).

**Neural Network Architecture:**
To solve this, treat it as multiple independent binary classification problems combined into one network.
*   The final output layer will have one neuron for each label being detected.
*   Use the **Sigmoid** activation function (not Softmax) for each of these output units.
*   The network will output independent probabilities for each label (e.g., $a_1=0.9$ for Car, $a_2=0.1$ for Bus, $a_3=0.8$ for Pedestrian).

---

## Advanced Optimization and Backpropagation

### 1. Advanced Optimization: The Adam Algorithm
Standard gradient descent uses a single global learning rate $\alpha$ for all parameters. However, some parameters might benefit from larger steps (if they consistently move in the same direction), while others need smaller steps (if they oscillate back and forth). 

*   **Adaptive Moment Estimation (Adam):** An advanced optimization algorithm that automatically adjusts the learning rate for *every single parameter* independently.
    *   If a parameter keeps moving in the same direction, Adam increases its learning rate to speed up training.
    *   If a parameter oscillates back and forth, Adam decreases its learning rate to smooth out the path toward the minimum.
*   **TensorFlow Implementation:**
    ```python
    import tensorflow as tf

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy()
    )
    ```
*   **Default Standard:** Adam has become the de facto standard optimizer in deep learning because it trains networks much faster and is more robust to the choice of the initial learning rate than traditional gradient descent.

---

### 2. Alternative Layer Types: Convolutional Layers
While dense layers connect every single neuron to all activations from the previous layer, other architectural types exist.
*   **Convolutional Layer:** Each neuron only looks at a restricted, local window of the input (e.g., a small region of an image or a short window of an EKG time-series signal).
*   **Benefits:**
    1.  Speeds up computation.
    2.  Requires less training data.
    3.  Less prone to overfitting.
*   **Convolutional Neural Network (CNN):** A neural network containing one or more convolutional layers, widely used in computer vision and signal processing.

---

### 3. The Computation Graph & Backpropagation
Modern frameworks like TensorFlow use **computation graphs** and **automatic differentiation (autodiff)** to compute derivatives without requiring manual calculus.

*   **Forward Propagation (Left to Right):** 
    *   Takes inputs ($X, W, B$) and flows step-by-step through nodes to compute intermediate values and ultimately the final cost $J$.
*   **Backpropagation (Right to Left):** 
    *   A right-to-left calculation that computes the partial derivative of the cost $J$ with respect to every intermediate node and parameter.
*   **Efficiency:** 
    *   If a graph has $N$ nodes and $P$ parameters, backpropagation computes all derivatives in roughly $N + P$ steps instead of $N \times P$ steps. This difference turns billions of operations into manageable fractions of a second for large networks.
