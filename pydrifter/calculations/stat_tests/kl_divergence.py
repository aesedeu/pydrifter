import dataclasses
import numpy as np
import pandas as pd
from scipy.stats import entropy
import pendulum

from pydrifter.calculations.stat import calculate_statistics
from pydrifter.base_classes.base_statistics import StatTestResult, BaseStatisticalTest


@dataclasses.dataclass
class KLDivergence(BaseStatisticalTest):
    control_data: np.ndarray
    treatment_data: np.ndarray
    feature_name: str = "UNKNOWN_FEATURE"
    bins: int = 50
    epsilon: float = 1e-8
    border_value: float = 0.1

    @property
    def __name__(self):
        return f"KL Divergence"

    def __call__(self) -> StatTestResult:
        data_min = min(self.control_data.min(), self.treatment_data.min())
        data_max = max(self.control_data.max(), self.treatment_data.max())
        bins = np.linspace(data_min, data_max, self.bins)

        p_hist, _ = np.histogram(self.control_data, bins=bins, density=True)
        q_hist, _ = np.histogram(self.treatment_data, bins=bins, density=True)

        p_hist += self.epsilon
        q_hist += self.epsilon

        p_hist /= p_hist.sum()
        q_hist /= q_hist.sum()

        kl_divergence = entropy(p_hist, q_hist)
        # kl_divergence = kl_div(self.control_data, self.treatment_data)

        if kl_divergence < self.border_value:
            conclusion = "OK"
        else:
            conclusion = "FAILED"

        control_data_statistics = calculate_statistics(self.control_data)
        treatment_data_statistics = calculate_statistics(self.treatment_data)

        statistics_result = pd.DataFrame(
            data={
                "test_datetime": [pendulum.now().to_datetime_string()],
                "feature_name": [self.feature_name],
                "feature_type": ["numerical"],
                "control_mean": [control_data_statistics["mean"]],
                "treatment_mean": [treatment_data_statistics["mean"]],
                "control_std": [control_data_statistics["std"]],
                "treatment_std": [treatment_data_statistics["std"]],
                "test_name": [self.__name__],
                "p_value": ["-"],
                "statistics": [kl_divergence],
                "conclusion": [conclusion],
            }
        )
        return StatTestResult(statistics_result=statistics_result)
