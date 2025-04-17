from abc import ABC
import pandas as pd
from ..preprocessing import ConfigMap
from ..statistics import calculate_statistics, TTest
from typing import Callable
from tabulate import tabulate
import warnings


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

        if len(self.data_treatment) < 1000 or len(self.data_control) < 1000:
            warnings.warn(f"data_control: {self.data_control.shape}")
            warnings.warn(f"data_treatment: {self.data_treatment.shape}")
            warnings.warn("Be careful with small amount of data. Some statistics may show incorrect results")

    def run(self, column_mapping: ConfigMap, features: list[str] = None):
        result_numerical = pd.DataFrame()

        if not features:
            features = column_mapping.numerical
        for test_name in self.tests:
            # print(test_name.__name__())
            for column in features:
                data_control_statistics = calculate_statistics(self.data_control[column])
                data_treatment_statistics = calculate_statistics(self.data_treatment[column])

                if column in column_mapping.numerical:

                    # p_value = test(data_1=self.data_control[column], data_2=self.data_treatment[column])
                    criterion = test_name(
                        data_1=self.data_control[column],
                        data_2=self.data_treatment[column],
                    )
                    p_value = criterion.run()

                    result_status = 'OK' if p_value >= 0.05 else "FAILED"

                    temp = pd.DataFrame(
                        data={
                            "feature_name": [column],
                            "control_mean": [data_control_statistics["mean"]],
                            "treatment_mean": [data_treatment_statistics["mean"]],
                            "control_std": [data_control_statistics["std"]],
                            "treatment_std": [data_treatment_statistics["std"]],
                            "stat_test": [criterion.__name__],
                            "p_value": [p_value],
                            "conclusion": [result_status],
                        }
                    )
                    result_numerical = pd.concat(
                        (result_numerical, temp), axis=0, ignore_index=True
                    )

        return tabulate(
            result_numerical, headers=result_numerical.columns, tablefmt="fancy_grid"
        )
