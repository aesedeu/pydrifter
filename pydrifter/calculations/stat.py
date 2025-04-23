import numpy as np


def mean_bootstrap(data: np.ndarray, size: int = 1_000):
    return np.array([float((np.random.choice(data, len(data), replace=True)).mean()) for _ in range(size)])

def calculate_statistics(data: np.array):
    return {
        "mean": data.mean(),
        "std": data.std(),
        "var": data.var()
    }
