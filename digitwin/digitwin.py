import os
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Any


class DigiTwinBase(ABC):

    def __init__(self, **kwargs):
        pass

    def step(self):
        pass

    def parse(self):
        pass

    @abstractmethod
    def itertimes(self, data: tuple):
        raise NotImplemented

    @abstractmethod
    def prepare_data(self, data: tuple):
        raise NotImplemented

    @abstractmethod
    def set_scenarios(self, data: dict, **kwargs) -> Any:
        raise NotImplemented

    @abstractmethod
    def inference(self):
        raise NotImplemented

    @abstractmethod
    def predict(self):
        raise NotImplemented

    @abstractmethod
    def performance(self, scenarios: dict) -> dict:
        raise NotImplemented

    @abstractmethod
    def visualize(self):
        raise NotImplemented

    @abstractmethod
    def export(self):
        raise NotImplemented


if __name__ == "__main__":

    pass

