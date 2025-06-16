# Contents of package:
# Classes and functions related to the basic functionality for TI-iDANSE.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import yaml
import pickle
import numpy as np
from dataclasses import dataclass, field

@dataclass
class Parameters:
    """A dataclass for simulation parameters."""
    K: int = 10  # number of nodes
    Qdglob: int = 1  # number of global desired sources
    Qnglob: int = 1  # number of global noise sources
    Qdloc: int = 0  # number of local desired sources
    Qnloc: int = 0  # number of local noise sources
    Mk: int = 3  # number of sensors per node
    D: int = 1  # number of target signal channels

    algos: list[str] = field(default_factory=lambda: [
        "centralized", "idanse", "tiidanse"
    ])  # list of algorithms to run

    seed: int = 42  # random number generator seed
    outputDir: str = ""  # path to output directory

    def __post_init__(self):
        np.random.seed(self.seed)
        # Validate parameters
        if self.Qdglob + self.Qnglob > self.Mk:
            raise ValueError("The sum of global desired and noise sources must not exceed the number of sensors per node.")
        self.M = self.K * self.Mk  # total number of sensors
        self.Qe = self.Qdglob + self.Qnglob  # effective number of channels exchanged between nodes

    def load_from_yaml(self, path: str):
        """Load parameters from a YAML file."""
        with open(path, 'r') as file:
            data = yaml.safe_load(file)
        for key, value in data.items():
            setattr(self, key, value)
        self.__post_init__()

    def export_to_pickle(self, name: str):	
        """Export parameters to a Pickle archive."""
        with open(f'{self.outputDir}/{name}.pkl', 'wb') as f:
            pickle.dump(self.__dict__, f)