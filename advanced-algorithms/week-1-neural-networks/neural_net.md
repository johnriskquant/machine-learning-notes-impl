## Table of Contents

* [1. Neural Networks and Deep Learning](#1-neural-networks-and-deep-learning)
  * [1.1 Motivation and History](#11-motivation-and-history)
  * [1.2 Why Are Neural Networks Taking Off Now?](#12-why-are-neural-networks-taking-off-now)
  * [1.3 Architecture of a Neural Network](#13-architecture-of-a-neural-network)
  * [1.4 Mathematics of a Neural Network Layer](#14-mathematics-of-a-neural-network-layer)
  * [1.5 Inference (Forward Propagation)](#15-inference-forward-propagation)
  * [1.6 Data Representation in TensorFlow vs. NumPy](#16-data-representation-in-tensorflow-vs-numpy)
  * [1.7 Building a Neural Network in Code (TensorFlow)](#17-building-a-neural-network-in-code-tensorflow)
* [2. Neural Network Implementation in Python: Forward Propagation](#2-neural-network-implementation-in-python-forward-propagation)
  * [2.1 Implementing a Dense Layer (With For-Loops)](#21-implementing-a-dense-layer-with-for-loops)
  * [2.2 Stringing Layers Sequentially](#22-stringing-layers-sequentially)
* [3. Matrix Multiplication and Vectorization](#3-matrix-multiplication-and-vectorization)
  * [3.1 Vector and Matrix Math Foundations](#31-vector-and-matrix-math-foundations)
  * [3.2 Vectorized Implementation of a Dense Layer](#32-vectorized-implementation-of-a-dense-layer)
* [4. Artificial Intelligence: AGI vs. ANI](#4-artificial-intelligence-agi-vs-ani)
  * [4.1 Artificial Narrow Intelligence (ANI)](#41-artificial-narrow-intelligence-ani)
  * [4.2 Artificial General Intelligence (AGI)](#42-artificial-general-intelligence-agi)
  * [4.3 The "One Learning Algorithm" Hypothesis](#43-the-one-learning-algorithm-hypothesis)

---

## 1. Neural Networks and Deep Learning

### 1.1 Motivation and History
* **Original Motivation:** Invented decades ago to write software that mimics how the human (or biological) brain learns and thinks.
* **Historical Timeline:**
  * Started in the 1950s, fell out of favor.
  * Gained popularity in the 1980s–1990s (used for handwritten digit recognition, zip codes, bank checks).
  * Fell out of favor in the late 1990s.
  * Resurged in 2005, rebranded as **Deep Learning**, revolutionizing applications like speech recognition, computer vision (ImageNet 2012), NLP, and more.
* **Biological Analogy (With a Caveat):**
  * *Biological Neuron:* Receives electrical impulses via inputs (*dendrites*), performs computation in the *cell body/nucleus*, and sends outputs via an *axon*.
  * *Artificial Neuron:* A simplified mathematical model that inputs numbers, does computation, and outputs a number.
  * *Caveat:* Today's deep learning is driven by engineering principles, not strict biological mimicry. We still don't fully understand how the human brain actually works.

---

### 1.2 Why Are Neural Networks Taking Off Now?
The rise of neural networks is largely due to two factors: **Data** and **Compute**.
* **Scale of Data:** As society digitized (internet, mobile phones), the amount of available data exploded.
* **Performance Plateaus:** Traditional ML algorithms (linear/logistic regression) plateau in performance even when fed massive amounts of data.
* **Neural Network Scaling:**
  * Small NN $\to$ Better than traditional ML.
  * Medium NN $\to$ Better performance.
  * Large NN $\to$ Performance keeps scaling upward with big data.
* **Hardware:** Faster computer processors, specifically the rise of GPUs (Graphics Processing Units), allowed researchers to train these massive networks efficiently.

---

### 1.3 Architecture of a Neural Network
Neural networks are built by grouping neurons into **layers**.

* **Input Layer (Layer 0):** The input features, represented as a vector $\vec{x}$.
* **Hidden Layer(s):** Groupings of neurons in the middle of the network. Called "hidden" because the training data does not explicitly provide the "correct" target values for these nodes.
* **Output Layer:** The final layer that produces the ultimate prediction (e.g., probability of a t-shirt being a top seller).
* **Activation ($a$):** A term from neuroscience. It refers to the output number a neuron computes and passes downstream.
* **Multi-Layer Perceptron (MLP):** A neural network with multiple hidden layers.

**Automated Feature Engineering:**
Instead of manually combining features (like width $\times$ depth = area), hidden layers automatically learn complex, intermediate features.
* *Example (Demand Prediction):* Layer 1 might take raw inputs (price, shipping, marketing) and automatically learn to represent concepts like "affordability" or "awareness" without being explicitly programmed to do so.
* *Example (Computer Vision):* Layer 1 learns short lines/edges $\to$ Layer 2 groups edges into parts (eyes, noses, car wheels) $\to$ Layer 3 groups parts into larger objects (faces, cars).

---

### 1.4 Mathematics of a Neural Network Layer
To distinguish between different layers and neurons, we use specific notation:
* **Superscript $[l]$:** Denotes a quantity associated with layer $l$.
* **Subscript $j$:** Denotes the $j$-th neuron (unit) in that layer.
* $g$: The activation function (e.g., the Sigmoid function).

**Computation for a single neuron $j$ in layer $l$:**
$$a_j^{[l]} = g\left( \vec{w}_j^{[l]} \cdot \vec{a}^{[l-1]} + b_j^{[l]} \right)$$
* $\vec{a}^{[l-1]}$ is the activation output vector from the *previous* layer.
* For the first hidden layer ($l=1$), the input is $\vec{a}^{[0]}$, which is just the input feature vector $\vec{x}$.

---

### 1.5 Inference (Forward Propagation)
Inference is the process of making a prediction by passing data through the network from left to right. It is called **Forward Propagation** because the activations propagate forward layer by layer.

*Example: Classifying an $8 \times 8$ image of a handwritten digit (64 input pixels, $x$):*
1. **Input:** $\vec{a}^{[0]} = \vec{x}$ (Vector of 64 numbers)
2. **Layer 1 (25 units):** $\vec{a}^{[1]} = g(W^{[1]} \cdot \vec{a}^{[0]} + \vec{b}^{[1]})$
3. **Layer 2 (15 units):** $\vec{a}^{[2]} = g(W^{[2]} \cdot \vec{a}^{[1]} + \vec{b}^{[2]})$
4. **Layer 3 (Output, 1 unit):** $a^{[3]} = g(W^{[3]} \cdot \vec{a}^{[2]} + b^{[3]})$
5. **Prediction:** Optionally threshold $a^{[3]}$ at $0.5$ to output $1$ or $0$.

---

### 1.6 Data Representation in TensorFlow vs. NumPy
TensorFlow is designed to handle massive datasets efficiently using matrices.
* **NumPy 1D Array:** `np.array([200, 17])` $\to$ Just a linear list of numbers (used heavily in Course 1).
* **TensorFlow / 2D Matrices:** TensorFlow prefers data structured as 2D arrays (matrices).
  * `np.array([[200, 17]])` $\to$ A $1 \times 2$ matrix (1 row, 2 columns). Known as a **row vector**.
  * `np.array([[200], [17]])` $\to$ A $2 \times 1$ matrix (2 rows, 1 column). Known as a **column vector**.
* **Tensors:** TensorFlow's internal data type for storing and operating on matrices efficiently. A tensor can be converted back to a standard NumPy array by calling `.numpy()` on it (e.g., `a1.numpy()`).

---

### 1.7 Building a Neural Network in Code (TensorFlow)
TensorFlow's Keras API provides a highly efficient way to string layers together using the `Sequential` framework.

**Explicit Layer-by-Layer Forward Propagation (Under the Hood):**
```python
x = np.array([[200.0, 17.0]]) # 1x2 Matrix

layer_1 = tf.keras.layers.Dense(units=3, activation='sigmoid')
a1 = layer_1(x)

layer_2 = tf.keras.layers.Dense(units=1, activation='sigmoid')
a2 = layer_2(a1)