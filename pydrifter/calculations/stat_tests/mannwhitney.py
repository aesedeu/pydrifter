import dataclasses
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import pendulum

from pydrifter.calculations.stat import calculate_statistics
from pydrifter.base_classes.base_statistics import StatTestResult, BaseStatisticalTest

@dataclasses.dataclass
class MannWhitney(BaseStatisticalTest):
    control_data: np.ndarray
    treatment_data: np.ndarray
    feature_name: str = "UNKNOWN_FEATURE"
    alpha: float = 0.05

    @property
    def __name__(self):
        return f"Mann-Whitney test"

    def __call__(self) -> StatTestResult:
        control_data_statistics = calculate_statistics(self.control_data)
        treatment_data_statistics = calculate_statistics(self.treatment_data)

        statistics, p_value = mannwhitneyu(self.control_data, self.treatment_data)

        result_status = "OK" if p_value >= self.alpha else "FAILED"

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
