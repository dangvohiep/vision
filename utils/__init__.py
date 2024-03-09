from collections import defaultdict
from typing import Any

class Accumulator:
    """
    A utility class for accumulating values for multiple metrics.
    """

    def __init__(self) -> None:
        self.__records: defaultdict[str, int] = defaultdict(int)

    def add(self, **kwargs: Any) -> None:
        """
        Add values to the accumulator.

        Parameters:
            kwargs: named metric and the value is the amount to add.
        """
        for metric, value in kwargs.items():
            # Each keyword argument represents a metric name and its value to be added
            self.__records[metric] += value
    
    def reset(self) -> None:
        """
        Reset the accumulator by clearing all recorded metrics.
        """
        self.__records.clear()

    def __getitem__(self, idx):
        return self.__records[idx]
