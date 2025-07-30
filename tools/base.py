# Contents of package:
# Classes and functions related to the basic functionality for TI-iDANSE.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import yaml
import pickle
import numpy as np
import scipy.signal as sig
from dataclasses import dataclass, field

@dataclass
class Parameters:
    """A dataclass for simulation parameters."""
    # WASN parameters
    K: int = 10  # number of nodes
    Mk: int = 3  # number of sensors per node
    graphDiameter: int = None  # diameter of the graph corresponding to the network

    # Acoustic scenario parameters
    c: float = 343.0  # speed of sound in m/s
    Qd: int = 1  # number of desired sources
    Qn: int = 1  # number of noise sources
    D: int = 1  # number of target signal channels
    observability: str = 'foss'  # "foss" for fully overlapping subspaces,
        # "poss" for partially overlapping subspaces
    possDiffuse: bool = False  # if True and observability is 'foss', the
        # noise sources are assumed only local, i.e., uncorrelated across nodes
    domain: str = 'time'  # domain of the signals, 'time', or 'time_complex', or 'wola'
    TDsteeringMats: str = 'random'  # type of steering matrices in time domain
        # 'random' for random matrices,
        # 'anechoic' for anechoic scenario (matrices entries are Green's function in free-field),
    roomLength: float = 10.0  # length of the room in meters
    roomWidth: float = 10.0   # width of the room in meters
    roomHeight: float = 3.0   # height of the room in meters
    t60: float = 0.0          # reverberation time in seconds
    minDistFromWall: float = 0.25  # minimum distance from the wall in meters
    minDistNodeSource: float = 0.25  # minimum distance from any node to a source in meters
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
    dynamics: str = 'static'  # type of acoustic scenario dynamics
        # 'static' for static scenario,
        # 'moving' for dynamic scenario with always-active sources and random source movements
        # 'switching' for dynamic scenario with sources that can appear and disappear
    movingEvery: float = 1.0  # time in seconds after which sources move in `moving` scenarios

    # Signals parameters
    fs: int = 16000  # sampling frequency
    T: float = 1.0  # duration of the signals in seconds
    N: int = field(init=False)  # number of samples, computed from fs and T
    selfNoiseFactor: float = 1e-6  # self-noise factor, used to scale the noise covariance
    noiseSigType: str = "random"  # random, babble, or ssn
    desSigType: str = "speech"  # speech or random
    speechDatabasePath: str = ""  # path to the speech database
    babbleDatabasePath: str = ""  # path to the babble database
    ssnDatabasePath: str = ""  # path to the SSN database
    speechFiles: list[str] = field(default_factory=list)  # list of speech files
    noiseFiles: list[str] = field(default_factory=list)  # list of noise files

    # Algorithm(s) parameters
    algos: list[str] = field(default_factory=lambda: [
        "centralized", "dmwf", "tidmwf"
    ])  # list of algorithms to run
    scmEstimation: str = 'oracle'  # type of SCM estimation to use
        # 'oracle' for perfect knowledge,
        # 'batch' for batch time-averaged,
        # 'online' for online
    maxDANSEiter: int = 100  # maximum number of iterations for DANSE
    mu: float = 1  # SDW-MWF factor
    gevd: bool = False  # if True, use GEV decomposition instead of regular MWF in estimation filters
    DANSEiterEveryXframes: int = 1  # DANSE iteration every X frames
    refNodeForTInorm: int = 0  # reference node for TI normalization
    dMWFalternating: bool = False  # if True, use alternating discovery/estimation steps in dMWF

    # VAD parameters
    useVAD: bool = False  # if True, update the SCMs based on a VAD decision
    noiseONOFFperiod: float = 0.5  # period of noise ON/OFF switching in seconds (only used if `useVAD` is True and desired signals are `random`)
    vadThreshold: float = 0.1  # threshold for VAD decision, relative to the energy of the latent desired signal
    vadSmoothingPeriod: float = 0.2  # smoothing period for VAD decision [s]

    # Online mode parameters
    frameDuration: float = 0.1  # length of the frame in seconds (for time-domain processing, otherwise using WOLA frames)
    beta: dict = field(default_factory=lambda: {'default': 0.99})  # forgetting factor for online SCM estimation

    # Debug
    singleLine: int = None  # if not None, only process this frequency line in WOLA domain
    unconstrainedRandomPositions: bool = False  # if True, allow random positions for sources
    wolaMixtures_viaTD: bool = False  # if True, build WOLA mixtures (final mic signals) via time-domain processing, then STFT. Otherwise, build directly in the WOLA domain via STFT of latent signals.

    seed: int = 42  # random number generator seed
    outputDir: str = ""  # path to output directory
    outputFilePath: str = ""  # path to output file
    suffix: str = ""  # output file suffix

    def __post_init__(self):
        np.random.seed(self.seed)
        self.N = int(self.fs * self.T)
        if self.scmEstimation == 'online':
            if self.domain == 'wola':
                self.frameDuration = (self.nfft - self.nhop) / self.fs  # frame length in seconds
            self.nFrames = int(np.floor(self.T / self.frameDuration))  # number of frames for online processing
            # TODO: improve that vvvv
            self.maxDANSEiter = 1  # one iteration per frame for online processing
        # Adjust beta's dictionary
        if 'default' in self.beta.keys():
            # Prepare a beta-string for later plotting
            self.betaString = ', '.join([f'$\\beta_\\text{{{k}}}={v}$' for k, v in self.beta.items()])
            self.beta = dict([(alg, self.beta.get(alg, self.beta['default'])) for alg in self.algos])
        # Number of positive frequencies in STFT
        self.nPosFreqs = self.nfft // 2 + 1 if self.singleLine is None else 1
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
                if 'tidmwf' in alg:
                    print(f'{alg} not implemented for partially overlapping subspaces.')
                    algs_to_remove.append(alg)
                    if 'tidanse' in self.algos:
                        print('...also removing TI-DANSE.')
                        algs_to_remove.append('tidanse')
            for alg in algs_to_remove:
                self.algos.remove(alg)
    
    def __str__(self):
        return str(self.__dict__)
    
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
    
    def get_stft(self, x):
        """
        Compute the STFT of the input signal.
        
        Parameters
        ----------
        x : np.ndarray
            Input signal, shape (N,) or (M, N) where N is the number of samples
            and M is the number of channels.

        Returns
        -------
        np.ndarray
            STFT of the input signal, shape (M, nPosFreqs, nFrames).
        """
        tmp = sig.stft(
            x,
            fs=self.fs,
            nperseg=self.nfft,
            noverlap=self.nfft - self.nhop,
            window=self.win,
            return_onesided=True,
        )[-1]
        if self.singleLine is not None:
            return tmp[..., [self.singleLine], :]
        else:
            return tmp
    
    def get_istft(self, X):
        if X.shape[-2] > 1:
            return sig.istft(
                X,
                fs=self.fs,
                nperseg=self.nfft,
                noverlap=self.nfft - self.nhop,
                window=self.win,
            )[1]
        else:
            # print("Warning: Only one frequency line is processed, ISTFT not applied.", end='\r')
            return X  # leave it as is if only one frequency line is processed


    def randmat(self, shape, makeComplex=True):
        """Generate a random matrix with given shape."""
        if self.domain == 'time' or not makeComplex:
            return np.random.randn(*shape)
        else:
            return np.random.randn(*shape) + 1j * np.random.randn(*shape)


def herm(x: np.ndarray) -> np.ndarray:
    """Hermitian transpose."""
    if x.ndim == 2:
        return x.conj().T
    elif x.ndim == 3:
        return x.conj().transpose(0, 2, 1)
