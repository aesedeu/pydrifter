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

    def run_data_health(self, remove_nan: bool = False, fill_nan: bool = False):
        """
        Check for missing values in the dataset and handle them based on specified flags.

        Parameters
        ----------
        remove_nan : bool, optional
            If True, rows with missing values will be removed from `self.data_treatment`.
        fill_nan : bool, optional
            If True, missing values will be filled using the corresponding column values from `self.data_control`.
            Numerical columns will be filled with the mean, categorical columns with the mode.

        Raises
        ------
        ValueError
            If both `remove_nan` and `fill_nan` are set to True.
        """
        if remove_nan and fill_nan:
            raise ValueError("Only one of 'remove_nan' or 'fill_nan' can be True at a time.")

        missing_counts = self.data_treatment.isna().sum()
        missing_with_values = missing_counts[missing_counts > 0]

        if missing_with_values.empty:
            print("✅ Данные в порядке: пропусков нет.")
            return True
        else:
            print("⚠️ Найдены пропуски в данных:")
            print(missing_with_values.to_dict())

            if remove_nan:
                self.data_treatment = self.data_treatment.dropna()
                print("🗑️ Строки с пропусками удалены.")

            elif fill_nan:
                for column in self.data_treatment.columns:
                    if self.data_treatment[column].isna().sum() > 0:
                        if self.data_treatment[column].dtype in ['float64', 'int64']:
                            fill_value = self.data_control[column].mean()
                        else:
                            fill_value = self.data_control[column].mode().iloc[0]
                        self.data_treatment.loc[:, column] = self.data_treatment[
                            column
                        ].fillna(fill_value)
                print("🧯 Пропуски заполнены значениями из контрольного набора.")
            else:
                return False

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
        show_result: bool = False,
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

        if show_result:
            return tabulate(
                result,
                headers=result_numerical.columns,
                tablefmt="pretty",
            )
        else:
            return result
