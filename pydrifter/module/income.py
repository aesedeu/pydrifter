from abc import ABC
import pandas as pd
from ..preprocessing import ConfigMap
from ..statistics import calculate_statistics, TTest
from typing import Callable
from tabulate import tabulate
from ..auxiliaries import *

warnings.showwarning = custom_warning


class Suite(ABC):

    def __init__(
        self,
        tests: list[Callable],
        data_control: pd.DataFrame,
        data_treatment: pd.DataFrame,
    ):
        self.tests = tests
        self.data_control = data_control
        self.data_treatment = data_treatment

        if (data_control.isna().sum().sum()) != 0:
            raise ValueError("Please replace NaN first in data_control")
        if (data_treatment.isna().sum().sum()) != 0:
            raise ValueError("Please replace NaN first in data_treatment")

        if len(self.data_treatment) < 1000 or len(self.data_control) < 1000:
            warnings.warn(f"data_control: {self.data_control.shape}")
            warnings.warn(f"data_treatment: {self.data_treatment.shape}")
            warnings.warn("Be careful with small amount of data. Some statistics may show incorrect results")

    def run(self, column_mapping: ConfigMap, features: list[str] = None, alpha: float = 0.05):
        assert alpha > 0 and alpha < 1, "Alpha (p-value) should be in [0;1] range"
        assert isinstance(features, (list, type(None))), "Features should be a python list with string values"

        result_numerical = pd.DataFrame()

        if not features:
            features = column_mapping.numerical
        for test_name in self.tests:
            for column in features:
                if column in column_mapping.numerical:
                    statistics_result = test_name(
                        data_1=self.data_control[column],
                        data_2=self.data_treatment[column],
                        feature_name=column,
                    ).run()
                    result_numerical = pd.concat(
                        (result_numerical, statistics_result), axis=0, ignore_index=True
                    )
                    result_numerical[
                        [
                            "control_mean",
                            "treatment_mean",
                            "control_std",
                            "treatment_std",
                            "statistics",
                            "p_value",
                        ]
                    ] = result_numerical[[
                        "control_mean",
                        "treatment_mean",
                        "control_std",
                        "treatment_std",
                        "statistics",
                        "p_value",
                    ]].round(4)

        return tabulate(
            result_numerical.sort_values("conclusion", ascending=False).reset_index(drop=True),
            headers=result_numerical.columns,
            tablefmt="pretty",
        )
