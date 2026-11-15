## Neural Networks and Deep Learning

### 1. Motivation and History
*   **Original Motivation:** Invented decades ago to write software that mimics how the human (or biological) brain learns and thinks.
*   **Historical Timeline:**
    *   Started in the 1950s, fell out of favor.
    *   Gained popularity in the 1980s–1990s (used for handwritten digit recognition, zip codes, bank checks).
    *   Fell out of favor in the late 1990s.
    *   Resurged in 2005, rebranded as **Deep Learning**, revolutionizing applications like speech recognition, computer vision (ImageNet 2012), NLP, and more.
*   **Biological Analogy (With a Caveat):** 
    *   *Biological Neuron:* Receives electrical impulses via inputs (*dendrites*), performs computation in the *cell body/nucleus*, and sends outputs via an *axon*.
    *   *Artificial Neuron:* A simplified mathematical model that inputs numbers, does computation, and outputs a number.
    *   *Caveat:* Today's deep learning is driven by engineering principles, not strict biological mimicry. We still don't fully understand how the human brain actually works.

---

### 2. Why are Neural Networks taking off now?
The rise of neural networks is largely due to two factors: **Data** and **Compute**.
*   **Scale of Data:** As society digitized (internet, mobile phones), the amount of available data exploded.
*   **Performance Plateaus:** Traditional ML algorithms (linear/logistic regression) plateau in performance even when fed massive amounts of data.
*   **Neural Network Scaling:** 
    *   Small NN $\to$ Better than traditional ML.
    *   Medium NN $\to$ Better performance.
    *   Large NN $\to$ Performance keeps scaling upward with big data.
*   **Hardware:** Faster computer processors, specifically the rise of GPUs (Graphics Processing Units), allowed researchers to train these massive networks efficiently.

---

### 3. Architecture of a Neural Network
Neural networks are built by grouping neurons into **layers**.

*   **Input Layer (Layer 0):** The input features, represented as a vector $\vec{x}$.
*   **Hidden Layer(s):** Groupings of neurons in the middle of the network. Called "hidden" because the training data does not explicitly provide the "correct" target values for these nodes.
*   **Output Layer:** The final layer that produces the ultimate prediction (e.g., probability of a t-shirt being a top seller).
*   **Activation ($a$):** A term from neuroscience. It refers to the output number a neuron computes and passes downstream. 
*   **Multi-Layer Perceptron (MLP):** A neural network with multiple hidden layers.

**Automated Feature Engineering:**
Instead of manually combining features (like width $\times$ depth = area), hidden layers automatically learn complex, intermediate features. 
*   *Example (Demand Prediction):* Layer 1 might take raw inputs (price, shipping, marketing) and automatically learn to represent concepts like "affordability" or "awareness" without being explicitly programmed to do so.
*   *Example (Computer Vision):* Layer 1 learns short lines/edges $\to$ Layer 2 groups edges into parts (eyes, noses, car wheels) $\to$ Layer 3 groups parts into larger objects (faces, cars).

---

### 4. Mathematics of a Neural Network Layer
To distinguish between different layers and neurons, we use specific notation:
*   **Superscript $[l]$:** Denotes a quantity associated with layer $l$.
*   **Subscript $j$:** Denotes the $j$-th neuron (unit) in that layer.
*   $g$: The activation function (e.g., the Sigmoid function).

**Computation for a single neuron $j$ in layer $l$:**
```math
a_j^{[l]} = g\left( \vec{w}_j^{[l]} \cdot \vec{a}^{[l-1]} + b_j^{[l]} \right)
```
*   $\vec{a}^{[l-1]}$ is the activation output vector from the *previous* layer.
*   For the first hidden layer ($l=1$), the input is $\vec{a}^{[0]}$, which is just the input feature vector $\vec{x}$.

---

### 5. Inference (Forward Propagation)
Inference is the process of making a prediction by passing data through the network from left to right. It is called **Forward Propagation** because the activations propagate forward layer by layer.

*Example: Classifying an $8 \times 8$ image of a handwritten digit (64 input pixels, $x$):*
1.  **Input:** $\vec{a}^{[0]} = \vec{x}$ (Vector of 64 numbers)
2.  **Layer 1 (25 units):** $\vec{a}^{[1]} = g(W^{[1]} \cdot \vec{a}^{[0]} + \vec{b}^{[1]})$
3.  **Layer 2 (15 units):** $\vec{a}^{[2]} = g(W^{[2]} \cdot \vec{a}^{[1]} + \vec{b}^{[2]})$
4.  **Layer 3 (Output, 1 unit):** $a^{[3]} = g(W^{[3]} \cdot \vec{a}^{[2]} + b^{[3]})$
5.  **Prediction:** Optionally threshold $a^{[3]}$ at $0.5$ to output $1$ or $0$.

---

### 6. Data Representation in TensorFlow vs. NumPy
TensorFlow is designed to handle massive datasets efficiently using matrices.
*   **NumPy 1D Array:** `np.array([200, 17])` $\to$ Just a linear list of numbers (used heavily in Course 1).
*   **TensorFlow / 2D Matrices:** TensorFlow prefers data structured as 2D arrays (matrices).
    *   `np.array([[200, 17]])` $\to$ A $1 \times 2$ matrix (1 row, 2 columns). Known as a **row vector**.
    *   `np.array([[200], [17]])` $\to$ A $2 \times 1$ matrix (2 rows, 1 column). Known as a **column vector**.
*   **Tensors:** TensorFlow's internal data type for storing and operating on matrices efficiently. A tensor can be converted back to a standard NumPy array by calling `.numpy()` on it (e.g., `a1.numpy()`).

---

### 7. Building a Neural Network in Code (TensorFlow)
TensorFlow's Keras API provides a highly efficient way to string layers together using the `Sequential` framework.

**Explicit Layer-by-Layer Forward Propagation (Under the Hood):**
```python
x = np.array([[200.0, 17.0]]) # 1x2 Matrix

layer_1 = tf.keras.layers.Dense(units=3, activation='sigmoid')
a1 = layer_1(x)

layer_2 = tf.keras.layers.Dense(units=1, activation='sigmoid')
a2 = layer_2(a1)
```

**The Sequential API (Standard Practice):**
Instead of manually passing activations, you define the architecture as a sequential pipeline:

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Define the model architecture
model = Sequential([
    Dense(units=25, activation='sigmoid', name='layer1'), # Layer 1
    Dense(units=15, activation='sigmoid', name='layer2'), # Layer 2
    Dense(units=1, activation='sigmoid')   # Output Layer
])

# 2. Compile the model (Prepare for training)
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
)

# 3. Train the model on data X and targets Y
model.fit(
    X,Y,            
    epochs=10,
)

# 4. Inference: Make predictions on new data
predictions = model.predict(X_new)
```

## Neural Network Implementation in Python: Forward Propagation

### 8. Implementing a Dense Layer (With For-Loops)
To build a neural network from scratch, we start by defining a single **Dense Layer**. This function takes the activations from the previous layer (`a_in`), the weights matrix (`W`), and the bias vector (`b`), and computes the activations for the current layer.

**Notation Convention:** 
*   Uppercase letters (e.g., $W$) denote matrices.
*   Lowercase letters (e.g., $a, b, w$) denote vectors or scalars.

**Python Implementation:**
    
    import numpy as np

    def dense(a_in, W, b):
        units = W.shape[1]             # Number of hidden units in this layer
        a_out = np.zeros(units)        # Initialize output array
        for j in range(units):
            w = W[:,j]                 # Extract the j-th column of weights
            z = w @ a_in + b[j]        # Dot product of weights and inputs + bias
            a_out[j] = g(z)            # Apply activation function (e.g., sigmoid)
        return a_out

### 9. Stringing Layers Sequentially
Once the `dense` function is defined, we can implement forward propagation for an entire neural network by passing the output of one layer directly as the input to the next layer.

**Python Implementation:**

    def sequential(x):
        a1 = dense(x, W1, b1)
        a2 = dense(a1, W2, b2)
        a3 = dense(a2, W3, b3)
        a4 = dense(a3, W4, b4)
        f_x = a4                       # Final output prediction
        return f_x

## Matrix Multiplication and Vectorization

### 1. Vector and Matrix Math Foundations
*   **Vector-Vector Dot Product:** 
    $$z = \vec{a} \cdot \vec{w} = a^T w$$
    This multiplies matching elements and sums them.
*   **Matrix Transpose ($A^T$):** Taking the columns of a matrix and laying them on their side to form rows.
*   **Matrix-Matrix Multiplication:**
    When multiplying $A^T \times W$, the element in row $i$ and column $j$ of the resulting matrix $Z$ is calculated by taking the dot product of the $i$-th row of $A^T$ and the $j$-th column of $W$.
    *   *Rule:* The inner dimensions must match (e.g., a $3 \times 2$ matrix multiplied by a $2 \times 4$ matrix yields a $3 \times 4$ matrix).

### 2. Vectorized Implementation of a Dense Layer
Using for-loops is computationally slow. Modern computers and GPUs are highly optimized to perform matrix multiplications in parallel. By representing inputs, weights, and biases as matrices, we can compute the entire layer's activations in a single step.

**Vectorized Python Implementation:**

    def dense(A_in, W, B):
        Z = np.matmul(A_in, W) + B     # Parallelized matrix multiplication
        A_out = g(Z)                   # Element-wise activation
        return A_out

**Implementation Notes (TensorFlow Convention):**
*   `np.matmul(A_in, W)` can also be written as `A_in @ W`.
*   In TensorFlow, individual training examples are laid out in rows within the matrix $X$ (which is why we use $A\_in$ directly rather than transposing it to $A^T$ in the code). 
*   Vectorization drastically reduces runtime for large neural networks, shifting execution time from hours down to minutes.

---

## Artificial Intelligence: AGI vs. ANI

### 1. Artificial Narrow Intelligence (ANI)
*   **Definition:** AI systems designed to do a single, narrow task extremely well.
*   **Examples:** Smart speakers, self-driving cars, web search, or specific industrial applications.
*   **Status:** Has seen tremendous progress and creates massive economic value today.

### 2. Artificial General Intelligence (AGI)
*   **Definition:** The dream of building AI systems that can do anything a typical human can do.
*   **Status:** Despite rapid progress in ANI, the path to AGI remains unclear. Progress in narrow tasks does not necessarily equal progress toward general intelligence.

### 3. The "One Learning Algorithm" Hypothesis
Neuroscience experiments suggest the biological brain might use a single, highly adaptable learning algorithm rather than hard-coded distinct functions.
*   **Neuroplasticity Experiments:** 
    *   If the auditory cortex is rewired to receive input from the eyes, it learns to *see*.
    *   If the somatosensory (touch) cortex is rewired to the eyes, it also learns to *see*.
    *   Humans can learn to "see" via voltages mapped to their tongues, or navigate using echolocation and magnetic haptic belts.
*   **Takeaway:** The brain is incredibly plastic and adapts to process whatever sensory data it receives. While modern artificial neural networks (like logistic regression units) are vastly oversimplified compared to biological neurons, replicating this adaptability remains a core inspiration for AGI research.
