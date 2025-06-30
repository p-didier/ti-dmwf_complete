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
    domain: str = 'time'  # domain of the signals, 'time' or 'wola'
    roomLength: float = 10.0  # length of the room in meters
    roomWidth: float = 10.0   # width of the room in meters
    roomHeight: float = 3.0   # height of the room in meters
    t60: float = 0.0          # reverberation time in seconds
    minDistFromWall: float = 0.25  # minimum distance from the wall in meters
    nodeRadius: float = 0.1  # radius of the node in meters
    onPlane: bool = True  # if True, all elements are on the same plane
    observabilityCriterion: str = 'raw_distance'  # criterion for computing observability
        # 'raw_distance': based on the distance between nodes and sources
        # 'hierarchical': starting from full observability, then removing observability connections starting from the largest node-source distance
    hierarchicalObsPruningThrs: float = 0.5  # for 'hierarchical' `observabilityCriterion`, threshold for pruning observability connections
        # if 0, no pruning is done
        # if 1, all possible observability connections are pruned, down to minimum criterion (observabilityCriterion = 'raw_distance')
        # intermediate values stop the pruning between 0 and 1.
    maxDistForObservability: float = 0.5  # minimum distance for observability in meters
    latentSNR: float = 0.0  # SNR between the latent desired signals and the latent noise signals in dB
    commDist: float = 1.5  # communication distance [m] (used at ad-hoc topology initialization, but `connectivity` plays a role)
    connectivity: float = 0.5  # amount of ad-hoc topology connectivity
    pruningStrategy: str = 'mst'  # tree-pruning strategy
        # 'star': star topology with updating node as root
        # 'mst': minimum spanning tree
        # 'mmut': MMUT pruning with updating node as root
        # 'line': line topology with updating node as root
    nfft: int = 1024  # number of FFT points
    nhop: int = 512  # number of samples in FFT hop
    win: str = 'hann'  # STFT window type

    # Signals parameters
    fs: int = 16000  # sampling frequency
    T: float = 1.0  # duration of the signals in seconds
    N: int = field(init=False)  # number of samples, computed from fs and T
    selfNoiseFactor: float = 1e-6  # self-noise factor, used to scale the noise covariance
    noiseSigType: str = "random"  # random or babble
    desSigType: str = "speech"  # speech or noise
    speechDatabasePath: str = ""  # path to the speech database
    babbleDatabasePath: str = ""  # path to the babble database
    speechFiles: list[str] = field(default_factory=list)  # list of speech files
    noiseFiles: list[str] = field(default_factory=list)  # list of noise files
    fileDuration: float = 5.0  # duration of the sound files in seconds
    selfNoiseFactor: float = 0.01  # self noise factor (as a fraction of the speech contributions)
    
    # Algorithm(s) parameters
    algos: list[str] = field(default_factory=lambda: [
        "centralized", "dmwf", "tidmwf"
    ])  # list of algorithms to run
    scmEstimation: str = 'oracle'  # type of SCM estimation to use
        # 'oracle' for perfect knowledge,
        # 'batch' for batch time-averaged,
        # 'online' for online
    maxDANSEiter: int = 100  # maximum number of iterations for DANSE

    # Metrics parameters
    metricsToCompute: list[str] = field(default_factory=lambda: [
        "msew",  # MSE between centralized and distributed filter coefficients
        "msed",  # MSE between estimated and true desired signals
    ])

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


def randmat(shape, makeComplex=True):
    """Generate a random matrix with given shape."""
    if makeComplex:
        return np.random.randn(*shape) + 1j * np.random.randn(*shape)
    else:
        return np.random.randn(*shape)