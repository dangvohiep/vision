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


class EarlyStopping:
    """
    A simple early stopping utility to terminate training when a monitored metric stops improving.

    Attributes:
        - patience (int): The number of epochs with no improvement after which training will be stopped.
        - tolerance (float): The minimum change in the monitored metric to qualify as an improvement,
        - considering the direction of the metric being monitored.
        - bestscore (float): The best score seen so far.
    """
    
    def __init__(self, patience: int, tolerance: float = 0.) -> None:
        """
        Initializes the EarlyStopping instance.
        
        Parameters:
            - patience (int): Number of epochs with no improvement after which training will be stopped.
            - tolerance (float): The minimum change in the monitored metric to qualify as an improvement. Defaults to 0.
        """
        self.patience: int = patience
        self.tolerance: float = tolerance
        self.bestscore: float = float('inf')
        self.__counter: int = 0

    def __call__(self, value: float) -> None:
        """
        Update the state of the early stopping mechanism based on the new metric value.

        Parameters:
            value (float): The latest value of the monitored metric.
        """
        # Improvement or within tolerance, reset counter
        if value <= self.bestscore + self.tolerance:
            self.bestscore = value
            self.__counter = 0

        # No improvement, increment counter
        else:
            self.__counter += 1

    def __bool__(self) -> bool:
        """
        Determine if the training process should be stopped early.

        Returns:
            bool: True if training should be stopped (patience exceeded), otherwise False.
        """
        return self.__counter >= self.patience

