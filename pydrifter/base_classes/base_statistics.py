import dataclasses
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import pendulum

@dataclasses.dataclass
class StatTestResult:
    dataframe: pd.DataFrame
    value: float
    conclusion: str = None


class BaseStatisticalTest(ABC):
    control_data: np.ndarray
    treatment_data: np.ndarray
    feature_name: str = "UNKNOWN_FEATURE"
    alpha: float = 0.1
    q: bool | float = False

    @property
    @abstractmethod
    def __name__(self) -> str:
        """Name of the statistical test."""
        pass

    @abstractmethod
    def __call__(self) -> StatTestResult:
        """Run statistical test and return result."""
        pass

    def _apply_quantile_cut(self, data: pd.Series | np.ndarray) -> pd.Series:
        """Apply quantile cut if self.q is set."""
        if not isinstance(data, pd.Series):
            data = pd.Series(data)
        return data[data < data.quantile(self.q)] if self.q else data

    def dataframe_report(
        self,
        control_mean,
        treatment_mean,
        control_std,
        treatment_std,
        test_name,
        quantile_cut,
        statistics,
        conclusion,
        feature_name: str = feature_name,
        feature_type: str = "unknown",
        p_value: str | float = "-",
        left_ci: float | str = "-",
        right_ci: float | str = "-"
    ) -> pd.DataFrame:
        statistics_result = pd.DataFrame(
            data={
                "test_datetime": [pendulum.now().to_datetime_string()],
                "feature_name": [feature_name],
                "feature_type": [feature_type],
                "control_mean": [control_mean],
                "treatment_mean": [treatment_mean],
                "control_std": [control_std],
                "treatment_std": [treatment_std],
                "quantile_cut": [quantile_cut],
                "test_name": [test_name],
                "p_value": [p_value],
                "left_ci": [left_ci],
                "right_ci": [right_ci],
                "statistics": [statistics],
                "conclusion": [conclusion],
            }
        )

        return statistics_result
