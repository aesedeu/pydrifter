import dataclasses
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

@dataclasses.dataclass
class StatTestResult:
    dataframe: pd.DataFrame
    value: float
    conclusion: str = None


class BaseStatisticalTest(ABC):
    control_data: np.ndarray
    treatment_data: np.ndarray
    feature_name: str
    alpha: float
    feature_name: str = "UNKNOWN_FEATURE"
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
