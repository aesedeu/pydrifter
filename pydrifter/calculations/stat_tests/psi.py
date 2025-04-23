import dataclasses
import numpy as np
import pandas as pd
import pendulum

from pydrifter.logger import create_logger
from pydrifter.calculations.stat import calculate_statistics
from pydrifter.base_classes.base_statistics import StatTestResult, BaseStatisticalTest

logger = create_logger(name="psi.py", level="info")

@dataclasses.dataclass
class PSI(BaseStatisticalTest):
    control_data: np.ndarray
    treatment_data: np.ndarray
    feature_name: str = "UNKNOWN_FEATURE"
    q: bool | float = False

    @property
    def __name__(self):
        return "Population Stability Index"

    def __call__(self) -> StatTestResult:
        control = self._apply_quantile_cut(self.control_data)
        treatment = self._apply_quantile_cut(self.treatment_data)

        control_data_statistics = calculate_statistics(control)
        treatment_data_statistics = calculate_statistics(treatment)

        bins = np.histogram_bin_edges(
                pd.concat([control, treatment], axis=0).values, bins="doane"
            )
        reference_percents = np.histogram(control, bins)[0] / len(control)
        current_percents = np.histogram(treatment, bins)[0] / len(treatment)

        np.place(
            reference_percents,
            reference_percents == 0,
            min(reference_percents[reference_percents != 0]) / 10**6
            if min(reference_percents[reference_percents != 0]) <= 0.0001
            else 0.0001,
        )
        np.place(
            current_percents,
            current_percents == 0,
            min(current_percents[current_percents != 0]) / 10**6
            if min(current_percents[current_percents != 0]) <= 0.0001
            else 0.0001,
        )

        psi_values = (reference_percents - current_percents) * np.log(
            reference_percents / current_percents
        )
        psi_value = np.sum(psi_values)

        if psi_value < 0.1:
            conclusion = "OK"
            logger.info(f"{self.__name__} for '{self.feature_name}'".ljust(50, ".") + " ✅ OK")
        else:
            conclusion = "FAILED"
            logger.info(f"{self.__name__} for '{self.feature_name}'".ljust(50, ".") + " ⚠️ FAILED")

        statistics_result = self.dataframe_report(
            feature_name=self.feature_name,
            feature_type="numerical",
            control_mean=control_data_statistics["mean"],
            treatment_mean=treatment_data_statistics["mean"],
            control_std=control_data_statistics["std"],
            treatment_std=treatment_data_statistics["std"],
            quantile_cut=self.q if self.q else False,
            test_name=self.__name__,
            statistics=psi_value,
            conclusion=conclusion,
        )

        return StatTestResult(
            dataframe=statistics_result, value=psi_value, conclusion=conclusion
        )
