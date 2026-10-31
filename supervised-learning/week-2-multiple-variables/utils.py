import numpy as np

def load_data(filepath: str)->tuple[np.ndarray, np.ndarray]:
    """
    Load the data from the txt file and return the features and labels as numpy arrays.

    Returns:
        tuple: A tuple containing two numpy arrays: features and labels.
    """

    data = np.loadtxt(filepath, delimiter=',')
    X = data[:, 0]  # All columns except the last one
    y = data[:, 1]     # Last column
    return X, y

def loan_data_multiple_lin_reg(filepath: str):
    data = np.loadtxt(filepath, delimiter=',')
    X = data[:, :2]  # All columns except the last one
    y = data[:, 2]     # Last column
    return X, y