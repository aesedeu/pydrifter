from abc import ABC
import pandas as pd
from ..preprocessing import DataConfig, GlobalConfig
from ..statistics import calculate_statistics, TTest
from typing import Callable, Type
from tabulate import tabulate
from ..auxiliaries import *
from ..logger import create_logger

warnings.showwarning = custom_warning
logger = create_logger(name="income.py", level="info")


class TableDriftChecker(ABC):
    """
    Abstract base class for performing drift checks between control and treatment datasets.

    This class validates the structural consistency between datasets, including:
    - Ensuring both are pandas DataFrames.
    - Verifying the number of columns, column names, and data types match.
    - Optionally performing a missing values health check upon initialization.

    Parameters
    ----------
    data_control : pd.DataFrame
        The baseline (control group) dataset.
    data_treatment : pd.DataFrame
        The dataset to be compared against the control (treatment group).
    data_config : DataConfig
        Configuration object containing feature metadata and missing value handling strategy.
    globl_config : Type[GlobalConfig], optional
        Global configuration class for setting common parameters and options. Defaults to `GlobalConfig`.

    Raises
    ------
    TypeError
        If either `data_control` or `data_treatment` is not a pandas DataFrame.
    UserWarning
        If either dataset contains fewer than 1000 observations.
    """

    def __init__(
        self,
        data_control: pd.DataFrame,
        data_treatment: pd.DataFrame,
        data_config: DataConfig,
        globl_config: Type[GlobalConfig] = GlobalConfig,
    ):
        self.data_control = data_control
        self.data_treatment = data_treatment
        self.data_config = data_config
        self.global_config = globl_config

        if not isinstance(self.data_control, pd.DataFrame):
            raise TypeError("`data_control` should be a pandas DataFrame")
        if not isinstance(self.data_treatment, pd.DataFrame):
            raise TypeError("`data_treatment` should be a pandas DataFrame")

        if len(self.data_treatment) < 1000 or len(self.data_control) < 1000:
            warnings.warn(f"data_control: {self.data_control.shape}")
            warnings.warn(f"data_treatment: {self.data_treatment.shape}")
            warnings.warn("Be careful with small amount of data. Some statistics may show incorrect results")

        self.run_data_health()

    def run_data_health(self, clean_data: bool = False):
        """
        Perform a health check on treatment and control datasets, validating their structure,
        data types, and presence of missing values. Optionally handles missing values based
        on the configured strategy.

        Checks performed:
        - Number of columns in both datasets must match.
        - Column names and their order must be identical.
        - Data types of corresponding columns must be the same.
        - Reports missing values in the treatment dataset.

        Parameters
        ----------
        clean_data : bool, optional
            If True, missing values in the treatment dataset will be handled according to
            `self.data_config.nan_strategy`:
            - "remove": removes rows with missing values.
            - "fill": fills missing values with the corresponding values from the control dataset
              (numerical columns with the mean, categorical columns with the mode).

        Raises
        ------
        ValueError
            If the number of columns or their names do not match between control and treatment datasets.
        TypeError
            If data types in corresponding columns differ.
        """

        # Number of cols checkup
        if self.data_control.shape[1] != self.data_treatment.shape[1]:
            raise ValueError(
                "Control and treatment datasets must have the same number of columns."
            )
        else:
            logger.info("Number of columns in datasets:".ljust(50, ".") + " ✅ OK")

        # Cols names
        if not all(self.data_control.columns == self.data_treatment.columns):
            raise ValueError(
                "Control and treatment datasets must have the same column names in the same order."
            )
        else:
            logger.info("Column names in datasets:".ljust(50, ".") + " ✅ OK")

        # Data types in cols
        control_dtypes = self.data_control.dtypes
        treatment_dtypes = self.data_treatment.dtypes
        mismatched_types = {
            col: (control_dtypes[col], treatment_dtypes[col])
            for col in self.data_control.columns
            if control_dtypes[col] != treatment_dtypes[col]
        }
        if mismatched_types:
            raise TypeError(f"Data type mismatch found in columns: {mismatched_types}")
        else:
            logger.info("Data types in datasets columns:".ljust(50, ".") + " ✅ OK")

        missing_counts = self.data_treatment.isna().sum()
        missing_with_values = missing_counts[missing_counts > 0]

        if missing_with_values.empty:
            logger.info("Missing values:".ljust(50, ".") + " ✅ OK")
        else:
            logger.info("Найдены пропуски в данных:".ljust(50, ".") + " ⚠️ FAILED")
            logger.info(missing_with_values.to_dict())

            if self.data_config.nan_strategy == "remove" and clean_data:
                self.data_treatment = self.data_treatment.dropna()
                logger.info("🗑️ Строки с пропусками удалены.")

            elif self.data_config.nan_strategy == "fill" and clean_data:
                for column in self.data_treatment.columns:
                    if self.data_treatment[column].isna().sum() > 0:
                        if self.data_treatment[column].dtype in ['float64', 'int64']:
                            fill_value = self.data_control[column].mean()
                        else:
                            fill_value = self.data_control[column].mode().iloc[0]
                        self.data_treatment.loc[:, column] = self.data_treatment[
                            column
                        ].fillna(fill_value)
                logger.info("🧯 Пропуски заполнены значениями из контрольного набора.")

    def __check_nan(self):
        """
        Validate that both control and treatment datasets contain no missing values.

        Raises
        ------
        ValueError
            If any missing (NaN) values are found in `data_control` or `data_treatment`.
        """
        if (self.data_control.isna().sum().sum()) != 0:
            raise ValueError("Please replace NaN first in data_control")
        if (self.data_treatment.isna().sum().sum()) != 0:
            raise ValueError("Please replace NaN first in data_treatment")

    def run_statistics(
        self,
        tests: list[Callable],
        features: list[str] = None,
        show_result: bool = False,
    ):
        """
        Run statistical tests on specified numerical features to compare control and treatment datasets.

        Parameters
        ----------
        tests : list of Callable
            A list of statistical test functions. Each function should accept arguments
            `data_1`, `data_2`, and `feature_name`, and implement a `.run()` method
            returning a pandas DataFrame with test results.
        features : list of str, optional
            List of numerical feature names to run the tests on.
            If None, features from `self.data_config.numerical` will be used.
        show_result : bool, optional, default=False
            If True, displays the results in a pretty-printed table format using `tabulate`.
            If False, returns the results as a pandas DataFrame.

        Returns
        -------
        Union[str, pd.DataFrame]
            Tabulated string with test results if `show_result` is True, otherwise a pandas DataFrame
            containing the statistical test results for each column.

        Raises
        ------
        TypeError
            If `features` is not a list of strings or None.
        ValueError
            If missing values (NaN) are detected in either dataset before running the tests.
        """
        if not isinstance(features, (list, type(None))):
            raise TypeError("`features` should be a Python list of string values or None")

        self.__check_nan()

        result_numerical = pd.DataFrame()
        if not features:
            features = self.data_config.numerical

        for test_name in tests:
            for column in features:
                if column in self.data_config.numerical:
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

                    logger.info(f"{test_name.__name__} for '{column}'".ljust(50, ".") + "SUCCEED")

        result = result_numerical.sort_values("conclusion", ascending=True).reset_index(drop=True)

        if show_result:
            return tabulate(
                result,
                headers=result_numerical.columns,
                tablefmt="pretty",
            )
        else:
            return result
