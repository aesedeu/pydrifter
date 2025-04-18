import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from scipy.stats import wasserstein_distance
from scipy.stats import entropy
from abc import ABC
import pendulum


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
                "test_datetime": [pendulum.now().to_datetime_string()],
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

    def __init__(
        self,
        data_1: np.ndarray,
        data_2: np.ndarray,
        feature_name: str = "UNKNOWN_FEATURE",
    ):
        self.data_1 = data_1
        self.data_2 = data_2
        self.feature_name = feature_name

    @property
    def __name__(self):
        return f"Wasserstein distance"

    def run(self):
        statistics = wasserstein_distance(self.data_1, self.data_2)
        data_1_statistics = calculate_statistics(self.data_1)
        data_2_statistics = calculate_statistics(self.data_2)

        statistics_result = pd.DataFrame(
            data={
                "test_datetime": [pendulum.now().to_datetime_string()],
                "feature_name": [self.feature_name],
                "control_mean": [data_1_statistics["mean"]],
                "treatment_mean": [data_2_statistics["mean"]],
                "control_std": [data_1_statistics["std"]],
                "treatment_std": [data_2_statistics["std"]],
                "test_name": [self.__name__],
                "p_value": ["-"],
                "statistics": [statistics],
                "conclusion": ["-"],
            }
        )
        return statistics_result


class KLDivergence(ABC):

    def __init__(
        self,
        data_1: np.ndarray,
        data_2: np.ndarray,
        feature_name: str = "UNKNOWN_FEATURE",
        bins: int = 50,
        epsilon: float = 1e-8,  # чтобы избежать деления на 0
    ):
        self.data_1 = data_1
        self.data_2 = data_2
        self.feature_name = feature_name
        self.bins = bins
        self.epsilon = epsilon

    @property
    def __name__(self):
        return f"KL Divergence"

    def run(self):
        # Общие границы для обоих распределений
        data_min = min(self.data_1.min(), self.data_2.min())
        data_max = max(self.data_1.max(), self.data_2.max())
        bins = np.linspace(data_min, data_max, self.bins)

        # Гистограммы (плотности)
        p_hist, _ = np.histogram(self.data_1, bins=bins, density=True)
        q_hist, _ = np.histogram(self.data_2, bins=bins, density=True)

        # Добавим небольшое значение, чтобы избежать деления на 0
        p_hist += self.epsilon
        q_hist += self.epsilon

        # Нормировка (на всякий случай)
        p_hist /= p_hist.sum()
        q_hist /= q_hist.sum()

        # KL Divergence (P || Q)
        kl_divergence = entropy(p_hist, q_hist)

        data_1_statistics = calculate_statistics(self.data_1)
        data_2_statistics = calculate_statistics(self.data_2)

        statistics_result = pd.DataFrame(
            data={
                "test_datetime": [pendulum.now().to_datetime_string()],
                "feature_name": [self.feature_name],
                "control_mean": [data_1_statistics["mean"]],
                "treatment_mean": [data_2_statistics["mean"]],
                "control_std": [data_1_statistics["std"]],
                "treatment_std": [data_2_statistics["std"]],
                "test_name": [self.__name__],
                "p_value": ["-"],
                "statistics": [kl_divergence],
                "conclusion": ["-"],
            }
        )
        return statistics_result


class PSI(ABC):
    def __init__(
        self,
        data_1: np.ndarray,
        data_2: np.ndarray,
        feature_name: str = "UNKNOWN_FEATURE",
        bins: int = 10,
    ):
        self.data_1 = data_1
        self.data_2 = data_2
        self.feature_name = feature_name
        self.bins = bins

    @property
    def __name__(self):
        return "Population Stability Index"

    def run(self):
        bin_edges = np.percentile(self.data_1, np.linspace(0, 100, self.bins + 1))
        data_1_counts, _ = np.histogram(self.data_1, bins=bin_edges)
        data_2_counts, _ = np.histogram(self.data_2, bins=bin_edges)

        data_1_percents = (
            data_1_counts / len(self.data_1) + 1e-8
        )  # избегаем деления на 0
        data_2_percents = data_2_counts / len(self.data_2) + 1e-8

        psi_values = (data_1_percents - data_2_percents) * np.log(
            data_1_percents / data_2_percents
        )
        psi_value = np.sum(psi_values)

        if psi_value < 0.1:
            conclusion = "OK"
        else:
            conclusion = "FAILED"

        statistics_result = pd.DataFrame(
            {
                "test_datetime": [pendulum.now().to_datetime_string()],
                "feature_name": [self.feature_name],
                "control_mean": [np.mean(self.data_1)],
                "treatment_mean": [np.mean(self.data_2)],
                "control_std": [np.std(self.data_1)],
                "treatment_std": [np.std(self.data_2)],
                "test_name": [self.__name__],
                "p_value": ["-"],
                "statistics": [psi_value],
                "conclusion": [conclusion],
            }
        )

        return statistics_result
