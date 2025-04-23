import dataclasses
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import pendulum

from pydrifter.logger import create_logger
from pydrifter.calculations.stat import mean_bootstrap, calculate_statistics
from pydrifter.base_classes.base_statistics import StatTestResult, BaseStatisticalTest

logger = create_logger(name="ttest.py", level="info")

@dataclasses.dataclass
class TTest(BaseStatisticalTest):
    control_data: np.ndarray
    treatment_data: np.ndarray
    var: bool = False
    alpha: float = 0.05
    feature_name: str = "UNKNOWN_FEATURE"
    q: bool | float = False

    @property
    def __name__(self):
        if self.var:
            return f"Student test (bootstrap mean)"
        else:
            return f"Welch's test (bootstrap mean)"

    def __call__(self) -> StatTestResult:
        control = self._apply_quantile_cut(self.control_data)
        treatment = self._apply_quantile_cut(self.treatment_data)

        control = mean_bootstrap(control)
        treatment = mean_bootstrap(treatment)

        control_data_statistics = calculate_statistics(control)
        treatment_data_statistics = calculate_statistics(treatment)

        statistics, p_value = ttest_ind(
            control,
            treatment,
            equal_var=self.var,
        )

        if p_value >= self.alpha:
            conclusion = "OK"
            logger.info(
                f"{self.__name__} for '{self.feature_name}'".ljust(50, ".") + " ✅ OK"
            )
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
            p_value=p_value,
            test_name=self.__name__,
            statistics=statistics,
            conclusion=conclusion,
        )
        return StatTestResult(
            dataframe=statistics_result, value=p_value, conclusion=conclusion
        )
