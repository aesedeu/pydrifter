from abc import ABC
from tabulate import tabulate
from typing import Union


class ConfigMap(ABC):

    def __init__(
            self,
            categorical: list[str],
            numerical: list[str],
            target: Union[str, None] = None,
        ):
        self.target = target
        self.categorical = categorical
        self.numerical = numerical

    def __repr__(self) -> str:
        return (
            f"ConfigMap(target={self.target}, "
            f"categorical={self.categorical}, "
            f"numerical={self.numerical})"
        )

    def show(self) -> None:
        """Print the configuration as a table."""
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
        print(tabulate(data, headers=["Parameter", "Value"], tablefmt="fancy_grid"))
