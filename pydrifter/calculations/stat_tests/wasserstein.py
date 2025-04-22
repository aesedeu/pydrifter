import dataclasses
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
import pendulum

from pydrifter.logger import create_logger
from pydrifter.calculations.stat import calculate_statistics
from pydrifter.base_classes.base_statistics import StatTestResult, BaseStatisticalTest

logger = create_logger(name="wasserstein.py", level="info")

@dataclasses.dataclass
class Wasserstein(BaseStatisticalTest):
    control_data: np.ndarray
    treatment_data: np.ndarray
    feature_name: str = "UNKNOWN_FEATURE"
    q: bool | float = False

    @property
    def __name__(self):
        return f"Wasserstein distance"

    def __call__(self) -> StatTestResult:
        # govnokod <3
        if self.q:
            control_data_q99 = self.control_data[
                self.control_data < self.control_data.quantile(self.q)
            ]
            treatment_data_q99 = self.treatment_data[
                self.treatment_data < self.treatment_data.quantile(self.q)
            ]

            control_data_statistics = calculate_statistics(control_data_q99)
            treatment_data_statistics = calculate_statistics(treatment_data_q99)

            wd_result = wasserstein_distance(control_data_q99, treatment_data_q99)

            norm = max(control_data_statistics["std"], 0.001)
            wd_result_norm = wd_result / norm
        else:
            control_data_statistics = calculate_statistics(self.control_data)
            treatment_data_statistics = calculate_statistics(self.treatment_data)

            wd_result = wasserstein_distance(self.control_data, self.treatment_data)

            norm = max(control_data_statistics["std"], 0.001)
            wd_result_norm = wd_result / norm

        if wd_result_norm < 0.1:
            conclusion = "OK"
            logger.info(f"{self.__name__} for '{self.feature_name}'".ljust(50, ".") + " ✅ OK")
        else:
            conclusion = "FAILED"
            logger.info(f"{self.__name__} for '{self.feature_name}'".ljust(50, ".") + " ⚠️ FAILED")

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
                "statistics": [wd_result_norm],
                "conclusion": [conclusion],
            }
        )
            
        return StatTestResult(
            dataframe=statistics_result, value=wd_result_norm, conclusion=conclusion
        )
