import dataclasses
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import pendulum

from pydrifter.logger import create_logger
from pydrifter.calculations.stat import calculate_statistics
from pydrifter.base_classes.base_statistics import StatTestResult, BaseStatisticalTest

logger = create_logger(name="kstest.py", level="info")

@dataclasses.dataclass
class KolmogorovSmirnov(BaseStatisticalTest):
    control_data: np.ndarray
    treatment_data: np.ndarray
    feature_name: str = "UNKNOWN_FEATURE"
    alpha: float = 0.05
    q: bool | float = False

    @property
    def __name__(self):
        return f"Kolmogorov-Smirnov test"

    def __call__(self) -> StatTestResult:
        control_data_statistics = calculate_statistics(self.control_data)
        treatment_data_statistics = calculate_statistics(self.treatment_data)

        statistics, p_value = ks_2samp(self.control_data, self.treatment_data)

        if p_value >= self.alpha:
            result_status = "OK"
            logger.info(f"{self.__name__} for '{self.feature_name}'".ljust(50, ".") + " ✅ OK")
        else:
            result_status = "FAILED"
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
                "p_value": [p_value],
                "statistics": [statistics],
                "conclusion": [result_status],
            }
        )
        return StatTestResult(dataframe=statistics_result, value=p_value)
