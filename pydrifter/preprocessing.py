from abc import ABC
from tabulate import tabulate
from typing import Union


class DataConfig(ABC):
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

    def __init__(
        self,
        categorical: list[str],
        numerical: list[str],
        nan_strategy: str = "fill",
        target: Union[str, None] = None
    ):
        if nan_strategy not in ["fill", "remove"]:
            raise TypeError(f"'nan_strategy' could be 'fill' or 'remove' only")

        self.target = target
        self.categorical = categorical
        self.numerical = numerical
        self.nan_strategy = nan_strategy

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
        ]
        return tabulate(data, headers=["Parameter", "Value"], tablefmt="fancy_grid")


class GlobalConfig():
    bootstrap_size = 50_000
