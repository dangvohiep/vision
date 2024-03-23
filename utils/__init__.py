import os
import pathlib
import time
from typing import Any, Dict
from collections import defaultdict

import torch
import torch.nn as nn

class Accumulator:
    """
    A utility class for accumulating values for multiple metrics.
    """

    def __init__(self) -> None:
        self.__records: defaultdict[str, float] = defaultdict(float)

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


class Timer:

    def __init__(self):
        self.__epoch_starts: Dict[int, float] = dict()
        self.__epoch_ends: Dict[int, float] = dict()
        self.__batch_starts: Dict[int, Dict[int, float]] = defaultdict(dict)
        self.__batch_ends: Dict[int, Dict[int, float]] = defaultdict(dict)

    def start_epoch(self, epoch: int) -> None:
        self.__epoch_starts[epoch] = time.time()

    def end_epoch(self, epoch: int) -> None:
        self.__epoch_ends[epoch] = time.time()

    def start_batch(self, epoch: int, batch: int = None) -> None:
        if batch is None:
            if self.__batch_starts[epoch]:
                batch = max(self.__batch_starts[epoch].keys()) + 1
            else:
                batch = 1
        self.__batch_starts[epoch][batch] = time.time()
    
    def end_batch(self, epoch: int, batch: int = None) -> None:
        if batch is None:
            if self.__batch_starts[epoch]:
                batch = max(self.__batch_starts[epoch].keys())
            else:
                raise RuntimeError(f"no batch has started")
        self.__batch_ends[epoch][batch] = time.time()
    
    def time_epoch(self, epoch: int) -> float:
        result = self.__epoch_ends[epoch] - self.__epoch_starts[epoch]
        if result > 0:
            return result
        else:
            raise RuntimeError(f"epoch {epoch} ends before starts")
    
    def time_batch(self, epoch: int, batch: int) -> float:
        result = self.__batch_ends[epoch][batch] - self.__batch_starts[epoch][batch]
        if result > 0:
            return result
        else:
            raise RuntimeError(f"batch {batch} in epoch {epoch} ends before starts")
        

class Logger:

    def __init__(self, logfile: os.PathLike):
        self.logfile = pathlib.Path(logfile)
        os.makedirs(name=self.logfile.parent, exist_ok=True)
        self._file = open(self.logfile, mode='w')

    def log(self, epoch: int, n_epochs: int, batch: int = None, n_batches: int = None, took: float = None, **kwargs):
        suffix = ', '.join([f'{metric}: {value:.3e}' for metric, value in kwargs.items()])
        prefix = f'Epoch {epoch}/{n_epochs} | '
        if batch is not None:
            prefix += f'Batch {batch}/{n_batches} | '
        if took is not None:
            prefix += f'Took {took:.2f}s | '
        logstring = prefix + suffix
        print(logstring)
        self._file.write(logstring + '\n')

    def __del__(self):
        self._file.close()


class CheckPointSaver:

    def __init__(self, dirpath: os.PathLike):
        self.dirpath = pathlib.Path(dirpath)
        os.makedirs(name=self.dirpath, exist_ok=True)

    def save(self, model: nn.Module, filename: os.PathLike) -> None:
        torch.save(obj=model, f=os.path.join(self.dirpath, filename))

