import dataclasses
import numpy as np
import pandas as pd
from scipy.stats import entropy
import pendulum

from pydrifter.logger import create_logger
from pydrifter.calculations.stat import calculate_statistics
from pydrifter.base_classes.base_statistics import StatTestResult, BaseStatisticalTest

logger = create_logger(name="kl_divergence.py", level="info")

@dataclasses.dataclass
class KLDivergence(BaseStatisticalTest):
    control_data: np.ndarray
    treatment_data: np.ndarray
    feature_name: str = "UNKNOWN_FEATURE"
    epsilon: float = 1e-8
    border_value: float = 0.1
    q: bool | float = False

    @property
    def __name__(self):
        return f"KL Divergence"

    def __call__(self) -> StatTestResult:
        if self.q:
            control_data_q99 = self.control_data[self.control_data < self.control_data.quantile(self.q)]
            treatment_data_q99 = self.treatment_data[self.treatment_data < self.treatment_data.quantile(self.q)]

            bins = np.histogram_bin_edges(pd.concat([control_data_q99, treatment_data_q99], axis=0).values, bins="doane")
            reference_percents = np.histogram(control_data_q99, bins)[0] / len(control_data_q99)
            current_percents = np.histogram(treatment_data_q99, bins)[0] / len(treatment_data_q99)
        else:
            bins = np.histogram_bin_edges(
                pd.concat([self.control_data, self.treatment_data], axis=0).values, bins="doane"
            )
            reference_percents = np.histogram(self.control_data, bins)[0] / len(self.control_data)
            current_percents = np.histogram(self.treatment_data, bins)[0] / len(self.treatment_data)

        np.place(
            reference_percents,
            reference_percents == 0,
            min(reference_percents[reference_percents != 0]) / 10**6
            if min(reference_percents[reference_percents != 0]) <= 0.0001
            else 0.0001,
        )

        np.place(
            current_percents,
            current_percents == 0,
            min(current_percents[current_percents != 0]) / 10**6
            if min(current_percents[current_percents != 0]) <= 0.0001
            else 0.0001,
        )

        kl_divergence = entropy(reference_percents, current_percents)

        if kl_divergence < self.border_value:
            conclusion = "OK"
            logger.info(f"{self.__name__} for '{self.feature_name}'".ljust(50, ".") + " ✅ OK")
        else:
            conclusion = "FAILED"
            logger.info(f"{self.__name__} for '{self.feature_name}'".ljust(50, ".") + " ⚠️ FAILED")

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
        return StatTestResult(
            dataframe=statistics_result, value=kl_divergence
        )
