import dataclasses
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import pendulum

from pydrifter.logger import create_logger
from pydrifter.calculations.stat import calculate_statistics
from pydrifter.base_classes.base_statistics import StatTestResult, BaseStatisticalTest

logger = create_logger(name="kstest.py", level="info")

@dataclasses.dataclass
class KolmogorovSmirnov(BaseStatisticalTest):
    control_data: np.ndarray
    treatment_data: np.ndarray
    feature_name: str = "UNKNOWN_FEATURE"
    alpha: float = 0.05
    q: bool | float = False

    @property
    def __name__(self):
        return f"Kolmogorov-Smirnov test"

    def __call__(self) -> StatTestResult:
        control = self._apply_quantile_cut(self.control_data)
        treatment = self._apply_quantile_cut(self.treatment_data)

        control_data_statistics = calculate_statistics(control)
        treatment_data_statistics = calculate_statistics(treatment)

        statistics, p_value = ks_2samp(control, treatment)

        if p_value >= self.alpha:
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
            p_value=p_value,
            test_name=self.__name__,
            statistics=statistics,
            conclusion=conclusion,
        )
        return StatTestResult(
            dataframe=statistics_result, value=p_value, conclusion=conclusion
        )

    def _ecdf(self, data):
        x = np.sort(data)
        y = np.arange(1, len(data) + 1) / len(data)
        return x, y

    def draw(self):
        statistics, p_value = ks_2samp(self.control_data, self.treatment_data)
        # Получаем ECDF
        x_x, x_y = self._ecdf(self.control_data)
        y_x, y_y = self._ecdf(self.treatment_data)

        # Объединяем все значения и считаем разницу ECDF на общей сетке
        all_values = np.sort(np.concatenate([self.control_data, self.treatment_data]))
        x_interp = np.searchsorted(x_x, all_values, side="right") / len(x_x)
        y_interp = np.searchsorted(y_x, all_values, side="right") / len(y_x)
        d = np.abs(x_interp - y_interp)
        max_idx = np.argmax(d)

        # Построение графика
        plt.figure(figsize=(10, 6))
        plt.step(x_x, x_y, label="control_data", color="dodgerblue", where="post")
        plt.step(y_x, y_y, label="treatment_data", color="orange", where="post")

        # Координаты точки максимального расстояния
        ks_x = all_values[max_idx]
        ecdf_x_val = x_interp[max_idx]
        ecdf_y_val = y_interp[max_idx]

        # Рисуем вертикальную линию KS-расстояния
        plt.vlines(
            ks_x,
            ymin=min(ecdf_x_val, ecdf_y_val),
            ymax=max(ecdf_x_val, ecdf_y_val),
            color="red",
            linestyle="--",
            label=f"KS distance = {statistics:.3f}, p-value = {p_value:.3f}",
        )

        # Подписи
        plt.title("ECDF KSTest")
        plt.xlabel("Value")
        plt.ylabel("Probability")
        plt.legend()
        plt.show()
