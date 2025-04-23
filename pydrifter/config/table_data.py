from abc import ABC
import dataclasses
from tabulate import tabulate
from typing import Union
import numpy as np


@dataclasses.dataclass
class TableConfig(ABC):
    """
    Configuration class for dataset feature types and missing value strategy.

    Parameters
    ----------
    categorical : List[str]
        List of categorical feature names.
    numerical : List[str]
        List of numerical feature names.
    nan_strategy : str, optional
        Strategy for handling missing values. Must be 'fill' or 'remove'. Default is 'fill'.
    target : str or None, optional
        Name of the target variable column, if present.
    """
    categorical: list[str]
    numerical: list[str]
    nan_strategy: str = "fill"
    target: Union[str] = "NOT DEFINED"
    quantiles_cut: bool | float = False

    if nan_strategy not in ["fill", "remove"]:
        raise TypeError(f"'nan_strategy' could be 'fill' or 'remove' only")

    if not isinstance(quantiles_cut, (bool, float)):
        raise TypeError("`quantiles_cut` should be bool or float type only")
    else:
        if isinstance(quantiles_cut, float):
            if quantiles_cut > 1.0 or quantiles_cut < 0:
                raise TypeError("`quantiles_cut` should be a in range [0;1]")

    def __repr__(self) -> str:
        data = [
            ["Target", self.target],
            [
                "Categorical Features",
                ", ".join(self.categorical) if self.categorical else "None",
            ],
            [
                "Numerical Features",
                ", ".join(self.numerical) if self.numerical else "None",
            ],
            ["NaN strategy", self.nan_strategy],
            ["quantiles_cut", self.quantiles_cut],
        ]
        return tabulate(data, headers=["Parameter", "Value"], tablefmt="fancy_grid")


class GlobalConfig():
    bootstrap_size = 50_000
