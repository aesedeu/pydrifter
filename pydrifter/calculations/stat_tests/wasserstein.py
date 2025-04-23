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
    alpha: float = 0.1
    q: bool | float = False

    @property
    def __name__(self):
        return f"Wasserstein distance"

    def __call__(self) -> StatTestResult:
        control = self._apply_quantile_cut(self.control_data)
        treatment = self._apply_quantile_cut(self.treatment_data)

        control_data_statistics = calculate_statistics(control)
        treatment_data_statistics = calculate_statistics(treatment)

        wd_result = wasserstein_distance(control, treatment)

        norm = max(control_data_statistics["std"], 0.001)
        wd_result_norm = wd_result / norm

        if wd_result_norm < self.alpha:
            conclusion = "OK"
            logger.info(f"{self.__name__} for '{self.feature_name}'".ljust(50, ".") + " ✅ OK")
        else:
            conclusion = "FAILED"
            logger.info(f"{self.__name__} for '{self.feature_name}'".ljust(50, ".") + " ⚠️ FAILED")

        statistics_result = self.dataframe_report(
            feature_name=self.feature_name,
            feature_type="numerical",
            control_mean=control_data_statistics["mean"],
            treatment_mean=treatment_data_statistics["mean"],
            control_std=control_data_statistics["std"],
            treatment_std=treatment_data_statistics["std"],
            quantile_cut=self.q if self.q else False,
            test_name=self.__name__,
            statistics=wd_result_norm,
            conclusion=conclusion,
        )

        return StatTestResult(
            dataframe=statistics_result, value=wd_result_norm, conclusion=conclusion
        )
