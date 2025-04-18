from abc import ABC
import pandas as pd
from ..preprocessing import DataConfig, GlobalConfig
from ..statistics import calculate_statistics, TTest
from typing import Callable, Type
from tabulate import tabulate
from ..auxiliaries import *

warnings.showwarning = custom_warning


class IncomeTableDrift(ABC):

    def __init__(
        self,
        tests: list[Callable],
        data_control: pd.DataFrame,
        data_treatment: pd.DataFrame,
        globl_config: Type[GlobalConfig] = GlobalConfig,
    ):
        self.tests = tests
        self.data_control = data_control
        self.data_treatment = data_treatment
        self.global_config = globl_config

        if not isinstance(self.data_control, pd.DataFrame):
            raise TypeError("`data_control` should be a pandas DataFrame")
        if not isinstance(self.data_treatment, pd.DataFrame):
            raise TypeError("`data_treatment` should be a pandas DataFrame")

        if len(self.data_treatment) < 1000 or len(self.data_control) < 1000:
            warnings.warn(f"data_control: {self.data_control.shape}")
            warnings.warn(f"data_treatment: {self.data_treatment.shape}")
            warnings.warn("Be careful with small amount of data. Some statistics may show incorrect results")

    def __clean_data(self):
        pass

    def run_data_health(self, clean: bool = True):
        pass

    def __check_nan(self):
        if (self.data_control.isna().sum().sum()) != 0:
            raise ValueError("Please replace NaN first in data_control")
        if (self.data_treatment.isna().sum().sum()) != 0:
            raise ValueError("Please replace NaN first in data_treatment")

    def run_statistics(
        self,
        data_config: DataConfig,
        features: list[str] = None,
        alpha: float = 0.05,
        show: bool = False,
    ):
        if not (0 < alpha < 1):
            raise ValueError("`alpha` (p-value threshold) should be in the (0, 1) range")

        if not isinstance(features, (list, type(None))):
            raise TypeError("`features` should be a Python list of string values or None")

        self.__check_nan()

        result_numerical = pd.DataFrame()
        if not features:
            features = data_config.numerical

        for test_name in self.tests:
            for column in features:
                if column in data_config.numerical:
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

                    print(f"{test_name.__name__} for '{column}'".ljust(50, ".") + "SUCCEED")

        result = result_numerical.sort_values("conclusion", ascending=True).reset_index(drop=True)

        if show:
            return tabulate(
                result,
                headers=result_numerical.columns,
                tablefmt="pretty",
            )
        else:
            return result
