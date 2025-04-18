from abc import ABC
from tabulate import tabulate
from typing import Union


class DataConfig(ABC):

    def __init__(
        self,
        categorical: list[str],
        numerical: list[str],
        target: Union[str, None] = None
    ):
        self.target = target
        self.categorical = categorical
        self.numerical = numerical

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
        ]
        return tabulate(data, headers=["Parameter", "Value"], tablefmt="fancy_grid")


class GlobalConfig():
    bootstrap_size = 50_000
