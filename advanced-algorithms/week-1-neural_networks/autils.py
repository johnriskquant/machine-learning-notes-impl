import numpy as np

def load_data(file_path):
    """
    Load data from the specified file path. 
    This function assumes that the data is stored in .npy format and that the features (X) and labels (y) are stored in separate files. The function loads the first 1000 samples of both X and y.  

    file_path: str
        The path to the .npy file containing the features (X). 
        The corresponding labels (y) are expected to be in a file with the same name but with 'X' replaced by 'y'.
    """
    X = np.load(file_path)  # Assuming X is stored in a .npy file
    y = np.load(file_path.replace('X', 'y'))  # Assuming y is stored in a separate file
    X = X[:1000]
    y = y[:1000]
    return X, y

def sigmoid(z):
    """
    Compute the sigmoid of z.

    z: array-like
        Input value or array of values.
    """
    return 1 / (1 + np.exp(-z))