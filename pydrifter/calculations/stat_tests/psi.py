import dataclasses
import numpy as np
import pandas as pd
import pendulum

from pydrifter.base_classes.base_statistics import StatTestResult, BaseStatisticalTest


@dataclasses.dataclass
class PSI(BaseStatisticalTest):
    control_data: np.ndarray
    treatment_data: np.ndarray
    feature_name: str = "UNKNOWN_FEATURE"

    @property
    def __name__(self):
        return "Population Stability Index"

    def __call__(self, q: bool = True) -> StatTestResult:
        if q:
            control_data_q99 = self.control_data[self.control_data < self.control_data.quantile(0.95)]
            treatment_data_q99 = self.treatment_data[self.treatment_data < self.treatment_data.quantile(0.95)]

            bins = np.histogram_bin_edges(pd.concat([control_data_q99, treatment_data_q99], axis=0).values, bins="doane")
            reference_percents = np.histogram(control_data_q99, bins)[0] / len(control_data_q99)
            current_percents = np.histogram(treatment_data_q99, bins)[0] / len(treatment_data_q99)
        else:
            bins = np.histogram_bin_edges(
                pd.concat([self.control_data, self.treatment_data], axis=0).values, bins="doane"
            )
            reference_percents = np.histogram(self.control_data, bins)[0] / len(self.control_data)
            current_percents = np.histogram(self.treatment_data, bins)[0] / len(self.treatment_data)

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
        else:
            conclusion = "FAILED"

        statistics_result = pd.DataFrame(
            {
                "test_datetime": [pendulum.now().to_datetime_string()],
                "feature_name": [self.feature_name],
                "feature_type": ["numerical"],
                "control_mean": [np.mean(self.control_data)],
                "treatment_mean": [np.mean(self.treatment_data)],
                "control_std": [np.std(self.control_data)],
                "treatment_std": [np.std(self.treatment_data)],
                "test_name": [self.__name__],
                "p_value": ["-"],
                "statistics": [psi_value],
                "conclusion": [conclusion],
            }
        )

        return StatTestResult(statistics_result=statistics_result)
