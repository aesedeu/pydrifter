import dataclasses
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import pendulum

from pydrifter.calculations.stat import mean_bootstrap, calculate_statistics
from pydrifter.base_classes.base_statistics import StatTestResult, BaseStatisticalTest


@dataclasses.dataclass
class TTest(BaseStatisticalTest):
    control_data: np.ndarray
    treatment_data: np.ndarray
    var: bool = False
    alpha: float = 0.05
    feature_name: str = "UNKNOWN_FEATURE"

    @property
    def __name__(self):
        if self.var:
            return f"Welch's test (sample mean)"
        else:
            return f"Welch's test (sample mean)"

    def __call__(self) -> StatTestResult:
        statistics, p_value = ttest_ind(
            mean_bootstrap(self.control_data),
            mean_bootstrap(self.treatment_data),
            equal_var=self.var,
        )
        result_status = "OK" if p_value >= self.alpha else "FAILED"

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
                "p_value": [p_value],
                "statistics": [statistics],
                "conclusion": [result_status],
            }
        )
        return StatTestResult(statistics_result=statistics_result)
