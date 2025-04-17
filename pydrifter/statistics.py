import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from abc import ABC


def mean_bootstrap(data: np.ndarray, size: int = 10_000):
    return np.array([float((np.random.choice(data, int(len(data) * 0.2))).mean()) for _ in range(size)])

def calculate_statistics(data: np.array):
    return {
        "mean": data.mean(),
        "std": data.std(),
        "var": data.var()
    }


class TTest(ABC):

    def __init__(self, data_1: np.ndarray, data_2: np.ndarray, var: bool = False):
        self.data_1 = data_1
        self.data_2 = data_2
        self.var = var

    @property
    def __name__(self):
        if self.var:
            return f"Student t-test"
        else:
            return f"Student t-test (Welch's test)"

    def run(self):
        _, p_value = ttest_ind(self.data_1, self.data_2, equal_var=self.var)
        return p_value
