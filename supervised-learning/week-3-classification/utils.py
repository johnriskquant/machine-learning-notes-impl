import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Union

def load_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load data from a text file.

    Parameters:
    file_path (str): The path to the text file.

    Returns:
    np.ndarray: The loaded data as a NumPy array.
    """

    data = np.loadtxt(file_path, delimiter=',')
    X = data[:, :2]  # Features
    y = data[:, 2]   # Labels
    return X, y

def sig(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
 
    return 1/(1+np.exp(-z))

def map_feature(X1: Union[float, np.ndarray], X2: Union[float, np.ndarray]) -> np.ndarray:
    """
    Feature mapping function to polynomial features    
    """
    X1 = np.atleast_1d(X1)
    X2 = np.atleast_1d(X2)
    degree = 6
    out = []
    for i in range(1, degree+1):
        for j in range(i + 1):
            out.append((X1**(i-j) * (X2**j)))
    return np.stack(out, axis=1)


def plot_data(X: np.ndarray, y: np.ndarray, pos_label: str = "y=1", neg_label: str = "y=0") -> None:
    positive = y == 1
    negative = y == 0
    
    # Plot examples
    plt.plot(X[positive, 0], X[positive, 1], 'k+', label=pos_label)
    plt.plot(X[negative, 0], X[negative, 1], 'yo', label=neg_label)
    
    
def plot_decision_boundary(w: np.ndarray, b: float, X: np.ndarray, y: np.ndarray) -> None:
    # Credit to dibgerge on Github for this plotting code
     
    plot_data(X[:, 0:2], y)
    
    if X.shape[1] <= 2:
        plot_x = np.array([min(X[:, 0]), max(X[:, 0])])
        plot_y = (-1. / w[1]) * (w[0] * plot_x + b)
        
        plt.plot(plot_x, plot_y, c="b")
        
    else:
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        
        z = np.zeros((len(u), len(v)))

        # Evaluate z = theta*x over the grid
        for i in range(len(u)):
            for j in range(len(v)):
                z[i,j] = sig(np.dot(map_feature(u[i], v[j])[0], w) + b)
        
        # important to transpose z before calling contour       
        z = z.T
        
        # Plot z = 0.5
        plt.contour(u,v,z, levels = [0.5], colors="g")