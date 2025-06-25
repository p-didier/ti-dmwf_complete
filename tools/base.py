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
    # WASN parameters
    K: int = 10  # number of nodes
    Mk: int = 3  # number of sensors per node
    graphDiameter: int = None  # diameter of the graph corresponding to the network

    # Acoustic scenario parameters
    Qd: int = 1  # number of desired sources
    Qn: int = 1  # number of noise sources
    D: int = 1  # number of target signal channels
    observability: str = 'foss'  # "foss" for fully overlapping subspaces,
        # "poss" for partially overlapping subspaces
    possDiffuse: bool = False  # if True and observability is 'foss', the
        # noise sources are assumed only local, i.e., uncorrelated across nodes

    # Signals parameters
    fs: int = 16000  # sampling frequency
    T: float = 1.0  # duration of the signals in seconds
    N: int = field(init=False)  # number of samples, computed from fs and T
    selfNoiseFactor: float = 1e-6  # self-noise factor, used to scale the noise covariance

    # Algorithm(s) parameters
    algos: list[str] = field(default_factory=lambda: [
        "centralized", "dmwf", "tidmwf"
    ])  # list of algorithms to run
    scmEstimation: str = 'oracle'  # type of SCM estimation to use
        # 'oracle' for perfect knowledge,
        # 'batch' for batch time-averaged,
        # 'online' for online

    seed: int = 42  # random number generator seed
    outputDir: str = ""  # path to output directory

    def __post_init__(self):
        np.random.seed(self.seed)
        self.N = int(self.fs * self.T)
        # Validate parameters
        if self.Qd + self.Qn > self.Mk:
            raise ValueError("The sum of global desired and noise sources must not exceed the number of sensors per node.")
        self.M = self.K * self.Mk  # total number of sensors
        self.Q = self.Qd + self.Qn  # total number of sources
        if self.observability == 'foss' and self.Q > self.Mk:
            raise ValueError("For fully overlapping subspaces, the number of global sources must not exceed the number of sensors per node.")
        if self.observability == 'poss':
            if self.Qd + self.Qn > self.Mk:
                raise ValueError("For partially overlapping subspaces, the total number of desired and noise sources must not exceed the number of sensors per node.")
                # ^^^ this is too strict, but we keep it for now
            algs_to_remove = []
            for alg in self.algos:
                if 'idanse' in alg:
                    print(f'{alg} not implemented for partially overlapping subspaces.')
                    algs_to_remove.append(alg)
            for alg in algs_to_remove:
                self.algos.remove(alg)

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