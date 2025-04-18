import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from scipy.stats import wasserstein_distance
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

    def __init__(
            self,
            data_1: np.ndarray,
            data_2: np.ndarray,
            var: bool = False,
            alpha: float = 0.05,
            feature_name: str = "UNKNOWN_FEATURE"
        ):
        self.data_1 = data_1
        self.data_2 = data_2
        self.var = var
        self.alpha = alpha
        self.feature_name = feature_name

    @property
    def __name__(self):
        if self.var:
            return f"Student t-test"
        else:
            return f"Student t-test (Welch's test)"

    def run(self):
        statistics, p_value = ttest_ind(self.data_1, self.data_2, equal_var=self.var)
        result_status = "OK" if p_value >= self.alpha else "FAILED"

        data_1_statistics = calculate_statistics(self.data_1)
        data_2_statistics = calculate_statistics(self.data_2)

        statistics_result = pd.DataFrame(
            data={
                "feature_name": [self.feature_name],
                "control_mean": [data_1_statistics["mean"]],
                "treatment_mean": [data_2_statistics["mean"]],
                "control_std": [data_1_statistics["std"]],
                "treatment_std": [data_2_statistics["std"]],
                "test_name": [self.__name__],
                "p_value": [p_value],
                "statistics": [statistics],
                "conclusion": [result_status],
            }
        )
        return statistics_result


class Wasserstein(ABC):
    def __init__(self, data_1: np.ndarray, data_2: np.ndarray):
        self.data_1 = data_1
        self.data_2 = data_2

    @property
    def __name__(self):
        return f"Wasserstein distance"

    def run(self):
        value = wasserstein_distance(self.data_1, self.data_2)
        return value
