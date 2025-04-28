from abc import ABC
import dataclasses
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pydrifter.config.table_data import TableConfig
from typing import Callable, Type
from tabulate import tabulate
from ..auxiliaries import *
from ..logger import create_logger

from pydrifter.base_classes.base_statistics import BaseStatisticalTest

warnings.showwarning = custom_warning
logger = create_logger(name="income.py", level="info")


@dataclasses.dataclass
class TableDrifter(ABC):
    data_control: pd.DataFrame
    data_treatment: pd.DataFrame
    data_config: TableConfig
    _results = None

    def __post_init__(self):
        if not isinstance(self.data_control, pd.DataFrame):
            raise TypeError("`data_control` should be a pandas DataFrame")
        if not isinstance(self.data_treatment, pd.DataFrame):
            raise TypeError("`data_treatment` should be a pandas DataFrame")
        if self.data_control.shape[1] != self.data_treatment.shape[1]:
            raise ValueError(f"Number of columns should be equal in control and treatment ({self.data_control.shape[1]} != {self.data_treatment.shape[1]})")

        selected_features = self.data_config.numerical + self.data_config.categorical
        self.data_control = self.data_control[selected_features]
        self.data_treatment = self.data_treatment[selected_features]

        if len(self.data_treatment) < 1000 or len(self.data_control) < 1000:
            warnings.warn(f"data_control: {self.data_control.shape}")
            warnings.warn(f"data_treatment: {self.data_treatment.shape}")
            warnings.warn("Be careful with small amount of data. Some statistics may show incorrect results")

    def run_data_health(self, clean_data: bool = False) -> None:
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

        # Missing values
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

    def __check_nan(self) -> None:
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
        tests: list[Type[BaseStatisticalTest]],
        show_result: bool = False,
    ) -> str | pd.DataFrame:
        """
        Run statistical tests on numerical and categorical features to compare control and treatment datasets.

        Parameters
        ----------
        tests : list of Type[BaseStatisticalTest]
            List of statistical test classes inheriting from `BaseStatisticalTest`.
            Each test should implement the `__call__()` method returning `StatTestResult`.

        show_result : bool, optional, default=False
            If True, returns a formatted string table using `tabulate`.
            If False, returns a pandas DataFrame with test results.

        Returns
        -------
        str or pandas.DataFrame
            Tabulated string of test results if `show_result` is True,
            otherwise a DataFrame containing the results of the statistical tests.

        Raises
        ------
        ValueError
            If missing values are found in the control or treatment datasets.
        """

        self.run_data_health()
        self.__check_nan()

        result_numerical = pd.DataFrame()

        features = self.data_config.numerical + self.data_config.categorical

        # Numerical tests
        for test_name in tests:
            for column in features:
                if column in self.data_config.numerical:
                    statistics_result = test_name(
                        control_data=self.data_control[column],
                        treatment_data=self.data_treatment[column],
                        feature_name=column,
                        q=self.data_config.quantiles_cut,
                    )()
                    result_numerical = pd.concat(
                        (result_numerical, statistics_result.dataframe),
                        axis=0,
                        ignore_index=True,
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

        result = result_numerical.sort_values("conclusion", ascending=True).reset_index(drop=True)

        self._results = result

        if show_result:
            print(tabulate(
                result,
                headers=result_numerical.columns,
                tablefmt="pretty",
            ))
        return result

    def draw(self, feature_name, quantiles: list[float] | None = None) -> None:
        if quantiles:
            if not isinstance(quantiles, list):
                raise TypeError("'quantiles' should be list or None")
            if quantiles[0] >= quantiles[1]:
                raise ValueError("Higher quantile should be higher than lower")
            if quantiles[0] < 0 or quantiles[1] > 1:
                raise ValueError("Quantiles should be in range [0;1]")

        if quantiles:
            control = self.data_control[
                (
                    self.data_control[feature_name]
                    > self.data_control[feature_name].quantile(quantiles[0])
                )
                & (
                    self.data_control[feature_name]
                    < self.data_control[feature_name].quantile(quantiles[1])
                )
            ][feature_name]
            sns.kdeplot(control, color="dodgerblue", label=f"Control (avg={control.mean():.2f})")

            test = self.data_treatment[
                (
                    self.data_treatment[feature_name]
                    > self.data_treatment[feature_name].quantile(quantiles[0])
                )
                & (
                    self.data_treatment[feature_name]
                    < self.data_treatment[feature_name].quantile(quantiles[1])
                )
            ][feature_name]
            sns.kdeplot(test, color="orange", label=f"Test (avg={test.mean():.2f})")
        else:
            sns.kdeplot(
                self.data_control[feature_name],
                color="dodgerblue",
                label=f"Control (avg={self.data_control[feature_name].mean():.2f})",
            )
            sns.kdeplot(
                self.data_treatment[feature_name],
                color="orange",
                label=f"Test (avg={self.data_treatment[feature_name].mean():.2f})",
            )

        plt.title(f"'{feature_name}' distribution")
        plt.legend()
        plt.show()

    def results(self):
        if self._results is not None:
            return self._results
        else:
            return "Not runned yet"
