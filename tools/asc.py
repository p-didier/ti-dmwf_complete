# Contents of package:
# Classes and functions related to the acoustic scenario for dMWF.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import copy
import time
import numpy as np
from .base import *
from tqdm import tqdm
import networkx as nx
import soundfile as sf
from pathlib import Path
import scipy.signal as sig
import anf_generator as anf
from resampy import resample
import pyroomacoustics as pra
from collections import deque
import matplotlib.pyplot as plt
from dataclasses import dataclass, field


SPEECH_DATABASE_NAME_DELIMITER = {
    'VCTK': 8
}

@dataclass
class StaticScenarioParameters:
    """A dataclass for static scenario parameters."""
    nodesPos: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    sensorsPos: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    speechSourcesPos: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    noiseSourcesPos: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    obsMat: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    rirs: list[np.ndarray] = field(default_factory=list)  # room impulse responses
    aMat: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))  # adjacency matrix
    trees: list = field(default_factory=list)  # list of TreeWASN objects
    Qkq: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))  # number of sources in common between nodes
    oQq: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))  # number of sources observed by each node
    Qdk: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))  # number of desired sources for each node

    def get_Qdims(self, c: Parameters):
        """Compute the number of sources in common between nodes."""
        if c.observability == 'foss': # or 1:  # DEBUG: always use 'foss' for now
            self.oQq = np.full(c.K, c.Q)
            self.Qkq = np.full((c.K, c.K), c.Q)
            self.Qdk = np.full(c.K, c.Qd)
            self.oQ = c.Q
        # elif c.observability == 'gls':
        #     raise NotImplementedError('TODO: implement GLS -- probably as constrained PODS -- and oQ should be possible to  compute based on the obsMat (i.e., number of sources observed by all nodes)')
        elif c.observability in ['poss', 'gls']:
            # Number of sources useful for fusion matrix computation for node q
            self.oQq = [0 for _ in range(c.K)]
            for k in range(c.K):
                for ii in range(c.Q):
                    if self.obsMat[k, ii] != 0 and np.sum(self.obsMat[:, ii]) > 1:
                        # If node k does observes source ii, and it is observed
                        # by at least one other node, then the number of sources
                        # in common is increased by one
                        self.oQq[k] += 1
            assert np.all(np.array(self.oQq) <= np.sum(self.obsMat, axis=1)), \
                "Number of sources in common exceeds number of sources observed by node."
            # Compute the number of sources in common between nodes k and q
            self.Qkq = np.zeros((c.K, c.K), dtype=int)
            for k in range(c.K):
                for q in range(c.K):
                    self.Qkq[k, q] = np.sum(self.obsMat[k, :] * self.obsMat[q, :])
            # Compute the number of desired sources for each node
            if c.nodeSpecificDANSEsourceEnum:
                self.Qdk = np.sum(self.obsMat[:, :c.Qd], axis=1)
            else:
                self.Qdk = np.full(c.K, c.Qd)
            self.oQ = len(np.where(self.obsMat.sum(axis=0) == c.K)[0])

        # ADJUST to number of microphones
        for k in range(c.K):
            self.Qkq[k, :] = [np.amin((c.Mk[k], q)) for q in self.Qkq[k, :]]
            self.oQq[k] = np.amin((c.Mk[k], self.oQq[k]))
            self.Qdk[k] = np.amin((c.Mk[k], self.Qdk[k]))

        # Compute and save compression factors
        c.sigDims = {
            'Qkq': self.Qkq,
            'oQq': self.oQq,
            'Qdk': self.Qdk,
        }
        c.CFs = {
            'danse': c.M / np.sum(self.Qdk),
            'dmwf': c.M / np.sum(self.oQq),
        }

        # Cache pre-computed selection matrices for dMWF
        self.Eqps = [[None for _ in range(c.K)] for _ in range(c.K)]
        for q in range(c.K):
            for p in range(c.K):
                if p == q:
                    continue
                # oQq_eff = np.amin((c.Mk[p], self.oQq[q]))
                oQq_eff = self.oQq[q]
                Qqp_eff = np.amin((c.Mk[p], self.Qkq[q, p]))
                self.Eqps[q][p] = c.init_full((c.Mk[p], oQq_eff), selection_matrix=True)
                self.Eqps[q][p][:Qqp_eff, :Qqp_eff] = np.eye(Qqp_eff)
                self.Eqps[q][p][Qqp_eff:, Qqp_eff:] = np.random.randn(
                    *(c.Mk[p] - Qqp_eff, oQq_eff - Qqp_eff)
                )

@dataclass
class SignalContainer:
    td: dict[np.ndarray] = field(default_factory=dict)
    wd: dict[np.ndarray] = field(default_factory=dict)
    
    def get_frame(self, l: int, domain='wd') -> dict[np.ndarray]:
        """Get the l-th signal frame."""
        if domain == 'td':
            return {
                key: val[:, l] if l > 0
                else np.zeros_like(val[:, l])
                for key, val in self.td.items()
            }
        elif domain == 'wd':
            return {
                key: val[..., l]  if l > 0
                else np.zeros_like(val[..., l])
                for key, val in self.wd.items()
            }
        else:
            raise ValueError("Domain must be 'td' or 'wd'.")


@dataclass
class Node(SignalContainer):
    """A dataclass for the node."""
    idx: int = None  # index of the node

    def init_signal_vectors(self, c: Parameters):
        self.td = {
            'y': np.zeros((c.Mk[self.idx], c.N)),
            's': np.zeros((c.Mk[self.idx], c.N)),
            'n': np.zeros((c.Mk[self.idx], c.N)),
            'sn': np.zeros((c.Mk[self.idx], c.N)),
            'sIndiv': np.zeros((c.Qd, c.Mk[self.idx], c.N)),
            'nIndiv': np.zeros((c.Qn, c.Mk[self.idx], c.N)),
        }
        if c.domain == 'wola':
            self.wd = {
                'y': np.zeros((c.Mk[self.idx], c.nPosFreqs, c.nFrames), dtype=complex),
                's': np.zeros((c.Mk[self.idx], c.nPosFreqs, c.nFrames), dtype=complex),
                'n': np.zeros((c.Mk[self.idx], c.nPosFreqs, c.nFrames), dtype=complex),
                'sn': np.zeros((c.Mk[self.idx], c.nPosFreqs, c.nFrames), dtype=complex)
            }

@dataclass
class TreeWASN:
    """A dataclass for the tree structure of the WASN."""
    nodesPos: dict[np.ndarray]  # positions of the nodes
    root: int  # root node
    aMat: np.ndarray  # adjacency matrix
    upNodes: list[list[int]]  # upstream nodes
    upNeighs: list[list[int]]  # upstream neighbors
    downNodes: list[int]  # downstream nodes
    downNeighs: list[int]  # downstream neighbors
    levels: list[list[int]]  # tree structure per depth level
    pathsToRoot: list[list[int]]  # path to root
    nq: list[list[int]]  # n(q) (as defined for TI-DANSE+)
    Qqup_k: list[int] = field(default_factory=list)  # $Q_{q\up k}$ as defined for TI-dMWF
        # i.e., number of sources observed by both node q and any node upstream of q
        # considering node `self.root` as the root.
    hMk: list[int] = field(default_factory=list)  # TI-dMWF dimension

    def is_upstream(self, up: int, down: int) -> bool:
        """Check if node `up` is upstream of node `down`."""
        if up == down:
            return True
        return up in self.upNodes[down]
    
    def is_downstream(self, down: int, up: int) -> bool:
        """Check if node `down` is downstream of node `up`."""
        # NB: This is not the same as checking if `down` is the (only)
        # downstream neighbor of `up`.
        if down == up:
            return True
        neigh = self.downNeighs[up]
        while neigh != down and neigh != self.root:
            neigh = self.downNeighs[neigh]
        return neigh == down
    
    def find_branch_origin(self, node: int) -> int:
        """Find the non-root origin of the branch that a given node belongs to."""
        if node == self.root:
            print("Node is the root, returning root as branch origin.")
            return self.root
        # Check if the node is upstream of the root
        if node in self.upNeighs[self.root]:
            return node
        return self.downNodes[node][-2]

    def find_path(self, start: int, end: int) -> list[int]:
        """Find the path from node `start` to node `end`."""
        if start == end:
            return [start]
        if not self.is_upstream(start, end):
            raise ValueError(f"Node {start} is not upstream of node {end}.")
        path = []
        current = start
        while current != end:
            path.append(current)
            current = self.downNeighs[current]
        path.append(end)
        return path

    def plot(self):
        """Plot the tree structure."""
        fig, axes = plt.subplots(1, 1)
        fig.set_size_inches(5.5, 5.5)
        # Plot the nodes
        for k, pos in self.nodesPos.items():
            x, y = np.mean(pos[:, 0]), np.mean(pos[:, 1])
            if k == self.root:
                # Highlight the root node
                axes.plot(x, y, 'ro', markersize=7)
                axes.text(
                    x + 0.1, y + 0.1, f'Root: {k}',
                    fontsize=10, color='red'
                )
            else:
                axes.plot(x, y, 'bo', markersize=5)
                axes.text(
                    x + 0.1, y + 0.1, f'{k}',
                    fontsize=10, color='blue'
                )
        # Plot the edges
        for k in range(self.aMat.shape[0]):
            for q in range(self.aMat.shape[1]):
                if self.aMat[k, q] > 0:
                    # Draw an edge from node k to node q
                    x1, y1 = np.mean(self.nodesPos[k][:, 0]), np.mean(self.nodesPos[k][:, 1])
                    x2, y2 = np.mean(self.nodesPos[q][:, 0]), np.mean(self.nodesPos[q][:, 1])
                    axes.plot([x1, x2], [y1, y2], 'k-', linewidth=0.5)
        axes.set_title(f'Tree structure of the WASN (root: {self.root})')
        axes.set_xlabel('X [m]')
        axes.set_ylabel('Y [m]')
        axes.grid()
        axes.set_aspect('equal', adjustable='box')
        fig.tight_layout()
        plt.show()


@dataclass
class AcousticScenario:
    """A dataclass for the acoustic scenario parameters."""
    cfg: Parameters
    nodes: list[Node] = field(default_factory=list)  # list of nodes
    latentDesired: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))  # latent desired signals
    latentNoise: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))  # latent noise signals
    obsMat: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))  # observability matrix
    Qkq: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))  # number of sources in common between nodes
    oQq: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))  # number of sources observed by each node
    scenarios: list[StaticScenarioParameters] = field(default_factory=list)  # list of static scenarios within the considered acoustic scenario

    def setup(self):
        c = self.cfg

        # Setup the acoustic scenario
        t0 = time.time()
        if c.domain == 'wola':
            out = self.setup_wola_domain()
        if 'time' in c.domain:
            if c.dynamics != 'static':
                raise ValueError(
                    "Dynamic scenarios are not supported in the time domain yet."
                )
            out = self.setup_time_domain()
        print(f">>> Acoustic scenario setup done in {time.time() - t0:.2f} s.")
        
        for asc in self.scenarios:
            asc.get_Qdims(c)

        # Print SNR at each node
        for k, node in enumerate(self.nodes):
            if c.domain == 'wola' and not c.wolaMixtures_viaTD:
                snr = 10 * np.log10(
                    np.mean(np.abs(node.wd['s']) ** 2) / np.mean(np.abs(node.wd['n']) ** 2)
                )
            else:
                snr = 10 * np.log10(
                    np.mean(np.abs(node.td['s']) ** 2) / np.mean(np.abs(node.td['n']) ** 2)
                )
            print(f"Node {k}: SNR = {snr:.2f} dB")

        return out
    
    def steermat_green_anechoic(self, sensorsPos: np.ndarray, sourcesPos: np.ndarray) -> np.ndarray:
        """Compute the steering matrix for the given sensors and sources positions."""
        # Compute the steering matrix
        Cmat = np.zeros((sensorsPos.shape[0], sourcesPos.shape[0]), dtype=complex)
        for ii in range(sourcesPos.shape[0]):
            for jj in range(sensorsPos.shape[0]):
                # Compute the distance between sensor jj and source ii
                dist = np.linalg.norm(sensorsPos[jj, :] - sourcesPos[ii, :])
                # Compute the steering vector
                Cmat[jj, ii] = 1 / dist # 1 / (4 * np.pi * dist) * np.exp(
                #     -1j * 2 * np.pi * c.fs * dist / c.c
                # )
        return Cmat

    def setup_time_domain(self):
        """Setup the acoustic scenario in the time domain."""
        c = self.cfg
        # Initialize matrices randomly
        if c.TDsteeringMats == 'random':
            # raise NotImplementedError('Random steering matrices not functioning with time-domain (trees needed for TI-DANSE+).')
            Amat = c.randmat((c.M, c.Qd))
            Bmat = c.randmat((c.M, c.Qn))
            nodesPos = c.randmat((c.K, 3))
            sensorsPos = c.randmat((c.M, 3))
            desSourcesPos = c.randmat((c.Qd, 3))
            noiSourcesPos = c.randmat((c.Qn, 3))
        elif c.TDsteeringMats == 'anechoic':
            # Define an anechoic scenario
            nodesPos, sensorsPos, desSourcesPos, noiSourcesPos = self.define_layout()
            Amat = self.steermat_green_anechoic(sensorsPos, desSourcesPos)
            Bmat = self.steermat_green_anechoic(sensorsPos, noiSourcesPos)
            if c.domain == 'time':
                Amat = np.abs(Amat)
                Bmat = np.abs(Bmat)
        
        slat = self.gen_latent_desired(n=c.Qd)
        nlat = self.get_latent_noise(n=c.Qn)
        pows = np.mean(np.abs(slat) ** 2, axis=1)
        pown = np.mean(np.abs(nlat) ** 2, axis=1)
        # Compute steering matrices
        if c.observability in ['poss', 'gls']:
            # Do not differentiate between global and local sources,
            # randomly generate observability pattern
            self.obsMat = np.zeros((c.K, c.Q))
            
            def inadequate(om):
                observedDesired = np.sum(om[:, :c.Qd], axis=1) > 0
                observedNoise = np.sum(om[:, c.Qd:], axis=1) > 0
                oneObserver = np.sum(om, axis=0) > 0 if c.observability == 'poss'\
                    else ((np.sum(om, axis=0) == 1) | (np.sum(om, axis=0) == c.K))  # <- GLS case
                return not np.all(observedDesired) or\
                    not np.all(observedNoise) or not np.all(oneObserver)
                    # not np.all(oneObserver)
            # Criterion for adequacy (PODS): at least one desired source and one noise
            # source must be observed by each node, and each source must be
            # observed by at least one node.
            while inadequate(self.obsMat):
                self.obsMat = np.random.randint(0, 2, (c.K, c.Q))
            if c.possDiffuse:
                # Make sure each noise source is only observed by one node at most
                for n in range(c.Qn):
                    idx = np.where(self.obsMat[:, c.Qd + n] == 1)[0]
                    if len(idx) > 1:
                        idx = np.random.choice(idx, 1, replace=False)
                        self.obsMat[:, c.Qd + n] = 0
                        self.obsMat[idx, c.Qd + n] = 1

            for k in range(c.K):
                for s in range(c.Qd):
                    if self.obsMat[k, s] == 0:
                        Amat[c.Mkc[k]:c.Mkc[k + 1], s] = 0
                for n in range(c.Qn):
                    if self.obsMat[k, c.Qd + n] == 0:
                        Bmat[c.Mkc[k]:c.Mkc[k + 1], n] = 0

        # Compute signals
        s = Amat @ slat
        n = Bmat @ nlat
        v = c.randmat((c.M, c.N)) * np.mean(pows) * c.selfNoiseFactor  # small self-noise
        n += v  # add self-noise to noise signal

        # Compute the SCMs
        if c.scmEstimation == 'oracle':
            # For oracle SCM estimation, we assume perfect knowledge of the
            # source and noise steering matrices
            Rsslat = np.diag(pows)
            Rnnlat = np.diag(pown)
            Rss = Amat @ Rsslat @ Amat.conj().T
            Rnn = Bmat @ Rnnlat @ Bmat.conj().T
            Rvv = np.eye(c.M) * np.mean(pows) * c.selfNoiseFactor  # small self-noise
            Rnn += Rvv  # add self-noise to noise SCM
            # Complete signal SCM
            Ryy = Rss + Rnn
        elif c.scmEstimation == 'batch':
            # Batch SCM estimation based on actual signals
            Rnn = n @ n.conj().T / c.N
            if c.noCrossCorrelation:
                Rss = s @ s.conj().T / c.N
                Ryy = Rss + Rnn
            else:
                Ryy = (s + n) @ (s + n).conj().T / c.N
                Rss = None
        elif c.scmEstimation == 'online':
            if not c.noCrossCorrelation:
                raise NotImplementedError('Not done yet for online time-domain')
            t0 = time.time()
            # Online SCM estimation
            betas = list(set(c.beta.values()))
            Rss = [{
                beta: c.randmat((c.M, c.M))
                if l == 0 else None for beta in betas
            } for l in range(c.nFrames)]
            Rnn = [{
                beta: c.randmat((c.M, c.M))
                if l == 0 else None for beta in betas
            } for l in range(c.nFrames)]
            nSamples = int(c.frameDuration * c.fs)
            for l in tqdm(range(1, c.nFrames), desc=f"Online SCM estimation"):
                idxBeg = l * nSamples
                idxEnd = (l + 1) * nSamples
                ssH = s[:, idxBeg:idxEnd] @ s[:, idxBeg:idxEnd].conj().T #/ nSamples ** 2
                nnH = n[:, idxBeg:idxEnd] @ n[:, idxBeg:idxEnd].conj().T #/ nSamples ** 2
                # Update the SCMs using the online estimation formula
                for beta in betas:
                    # Update the SCMs using the online estimation formula
                    Rss[l][beta] = beta * Rss[l - 1][beta] + (1 - beta) * ssH
                    Rnn[l][beta] = beta * Rnn[l - 1][beta] + (1 - beta) * nnH
            Ryy = [
                {beta: Rss[l][beta] + Rnn[l][beta] for beta in betas}
                for l in range(c.nFrames)
            ]
            print(f"Online SCM estimation done in {time.time() - t0:.2f} s.")
        else:
            raise ValueError(f"Unknown SCM estimation method: {c.scmEstimation}")
        
        self.scenarios = [StaticScenarioParameters(
            nodesPos=np.zeros((c.K, 3)),  # Placeholder for nodes positions
            sensorsPos=np.zeros((c.M, 3)),  # Placeholder for sensors positions
            speechSourcesPos=np.zeros((c.Qd, 3)),  # Placeholder for speech sources positions
            noiseSourcesPos=np.zeros((c.Qn, 3)),  # Placeholder for noise sources positions
            obsMat=self.obsMat,
            rirs=[],  # No RIRs in time domain
            aMat=np.zeros((c.K, c.K)),  # Placeholder for adjacency matrix
            trees=[],  # No trees in time domain
            Qkq=self.Qkq,
            oQq=self.oQq
        )]


        # Non-fully connected network
        aMat, commDist = get_adjacency_matrix(c, nodesPos)
        # Ensure target degree of connectivity
        aMat = self.set_target_connectivity(aMat, c.connectivity)
        obsMat = self.get_observability_matrix(
            nodesPos, desSourcesPos, noiSourcesPos
        )  # node x source
        self.scenarios[0].trees = [
            self.tree_pruning(
                k, sensorsPos,
                aMat, obsMat,
                method=c.pruningStrategy
            )
            for k in range(c.K)
        ]


        return Ryy, Rss, Rnn, s, n
    
    def set_target_connectivity(self, aMat, targetConn=None, rng=np.random.RandomState()):
        """Ensure target degree of connectivity in non-fully connected network."""
        c = self.cfg  # alias for convenience
        n1s_fc = c.K * (c.K - 1)  # number of 1's in adjacency matrix for full connectivity
        n1s_mc = 2 * c.K  # number of 1's in adjacency matrix for minimum connectivity
        currConn = compute_connectivity(c, aMat)
        flagPrint = True
        if targetConn is not None:
            rngConn = np.random.RandomState(rng.randint(0, 1000))  # for reproducibility
            while currConn > targetConn and currConn - 2 / (n1s_fc - n1s_mc) >= targetConn:
                if flagPrint:
                    flagPrint = False
                # Remove two connections to make the graph less connected
                # (ensuring that the graph remains connected)
                k1, k2 = rngConn.choice(c.K, 2, replace=False)
                if aMat[k1, k2] == 1:
                    aMat[k1, k2] = 0
                    aMat[k2, k1] = 0
                # Check if the graph is still connected
                while not np.linalg.matrix_power(aMat, c.K).all():
                    aMat[k1, k2] = 1
                    aMat[k2, k1] = 1
                    k1, k2 = rngConn.choice(c.K, 2, replace=False)
                    if aMat[k1, k2] == 1:
                        aMat[k1, k2] = 0
                        aMat[k2, k1] = 0
                currConn = compute_connectivity(c, aMat)
            while currConn < targetConn:
                if flagPrint:
                    flagPrint = False
                # Add two connections to make the graph more connected
                k1, k2 = rngConn.choice(c.K, 2, replace=False)
                if aMat[k1, k2] == 0:
                    aMat[k1, k2] = 1
                    aMat[k2, k1] = 1
                currConn = compute_connectivity(c, aMat)
        return aMat
    
    def setup_wola_domain(self):
        c = self.cfg

        # Set up latent signals
        self.latentDesired = self.gen_latent_desired(n=c.Qd)
        self.latentNoise = self.get_latent_noise(n=c.Qn)
        # Scale the latent noise signals to match the target SNR
        if c.latentSNR is not None:
            # Scale the latent noise signals to match the target SNR
            vad = self.compute_vad(np.sum(self.latentDesired, axis=0))
            nPower = np.mean(np.abs(self.latentNoise[:, vad]) ** 2)
            sPower = np.mean(np.abs(self.latentDesired[:, vad]) ** 2)
            if nPower > 0 and sPower > 0:
                currentSNR = 10 * np.log10(sPower / nPower)
                scalingFactor = 10 ** ((c.latentSNR - currentSNR) / 20)
                self.latentNoise /= scalingFactor
        
        # Prepare Node containers
        self.nodes = [Node(idx=k) for k in range(c.K)]
        for node in self.nodes:
            node.init_signal_vectors(c)

        # Define the acoustic scenario
        if c.dynamics == 'static' or c.scmEstimation != 'online':
            print(f"Defining base acoustic scenario...")
            room = self.setup_wola_domain_static()
            print("Base acoustic scenario defined.")
            if 1:
                self.plot()
        elif c.dynamics == 'moving' and c.scmEstimation == 'online':
            # Dynamic scenario with always-active sources and random source movements
            nScenarios = np.floor(c.T / c.movingEvery).astype(int)
            print(f"Setting up {nScenarios} dynamic scenarios with different sources/nodes positions.")
            
            if c.fixedObservabilities:
                omFixed = self.get_observability_matrix()  # Get the fixed observability matrix
            else:
                omFixed = None

            print(f"Defining base acoustic scenario...")
            room = self.setup_wola_domain_static(
                idxStart=0,
                idxEnd=int(c.movingEvery * c.fs) - 1,
                omFixed=omFixed
            )
            print("Base acoustic scenario defined.")
            for ii in range(1, nScenarios):
                print(f"Setting up scenario {ii + 1}/{nScenarios}...")
                idxStart = ii * int(c.movingEvery * c.fs)
                idxEnd = idxStart + int(c.movingEvery * c.fs) - 1
                # Setup the static scenario
                self.shuffle_scenario(idxStart, idxEnd)

        if c.diffuseSNR is not None and c.diffuseSNR > -50:
            # Compute and add diffuse noise
            params = anf.CoherenceMatrix.Parameters(
                mic_positions=self.scenarios[0].sensorsPos,
                sc_type="spherical",
                sample_frequency=c.fs,
                nfft=c.nfft,
            )
            input_signals = np.random.randn(c.M, int(c.T * params.sample_frequency))
            t0 = time.time()
            print("Generating diffuse noise...")
            diffuseNoise, _, _ = anf.generate_signals(
                input_signals,
                params,
                decomposition='evd',
                processing='balance+smooth'
            )
            print(f"Diffuse noise generated in {time.time() - t0:.2f} s.")
            # Normalize the diffuse noise
            diffuseNoise /= np.linalg.norm(diffuseNoise, axis=0, keepdims=True)
            # Adapt SNR of diffuse noise to the target SNR
            if c.diffuseSNR is not None:
                # Scale the diffuse noise signals to match the target SNR
                nPower = np.mean(np.abs(diffuseNoise) ** 2)
                sPower = np.mean(np.abs(self.latentDesired) ** 2)
                if nPower > 0 and sPower > 0:
                    currentSNR = 10 * np.log10(sPower / nPower)
                    scalingFactor = 10 ** ((c.diffuseSNR - currentSNR) / 20)
                    diffuseNoise *= scalingFactor
            
            if not c.wolaMixtures_viaTD:
                # If we don't pass by the TD, convert diffuse noise to STFT domain
                diffuseNoise = c.get_stft(diffuseNoise)

            for k in range(c.K):
                if c.wolaMixtures_viaTD:
                    self.nodes[k].td['n'] += diffuseNoise[c.Mkc[k]:c.Mkc[k + 1], ...]
                else:
                    self.nodes[k].wd['n'] += diffuseNoise[c.Mkc[k]:c.Mkc[k + 1], ...]

        # Self-noise addition (constant, independent of dynamics)
        pows = np.mean(np.abs(self.latentDesired) ** 2, axis=1)
        for k in range(c.K):
            # Generate self-noise at correct SNR
            sn = c.randmat((c.Mk[k], c.N), makeComplex=False) * np.mean(pows) * c.selfNoiseFactor
            if c.wolaMixtures_viaTD:
                self.nodes[k].td['sn'] = sn
                self.nodes[k].td['n'] += self.nodes[k].td['sn']
                self.nodes[k].td['y'] = self.nodes[k].td['n'] + self.nodes[k].td['s']
            else:
                snSTFT = c.get_stft(sn)
                self.nodes[k].wd['sn'] = snSTFT
                self.nodes[k].wd['n'] += self.nodes[k].wd['sn']
                self.nodes[k].wd['y'] = self.nodes[k].wd['n'] + self.nodes[k].wd['s']

        # Compute per sensor per source signal powers
        sigPows = np.zeros((c.M, c.Q))
        for k in range(c.K):
            for q in range(c.Qd):
                if self.scenarios[0].obsMat[k, q] == 1:
                    if c.wolaMixtures_viaTD:
                        sig = self.nodes[k].td['sIndiv'][q, :, :]
                    else:
                        sig = self.nodes[k].wd['sIndiv'][q, :, :]
                    sigPows[c.Mkc[k]:c.Mkc[k + 1], q] = np.mean(np.abs(sig) ** 2, axis=1)
            for q in range(c.Qn):
                if self.scenarios[0].obsMat[k, c.Qd + q] == 1:
                    if c.wolaMixtures_viaTD:
                        sig = self.nodes[k].td['nIndiv'][q, :, :]
                    else:
                        sig = self.nodes[k].wd['nIndiv'][q, :, :]
                    sigPows[c.Mkc[k]:c.Mkc[k + 1], c.Qd + q] = np.mean(np.abs(sig) ** 2, axis=1)
        # Make relative to max
        sigPows /= np.max(sigPows)
        # Only keep reference sensor of each node
        sigPows = sigPows[c.Mkc[:-1], :]
        # Compute self-noise power at each sensor
        P_self = np.ones(c.K) * np.mean(pows) * c.selfNoiseFactor

        obsMat = compute_observability_matrix(
            P_ms=sigPows,
            P_self=P_self,
        )

        print("Acoustic environment generated successfully, computing SCMs...")
        # Compute SCMs (from steering matrices if oracle, from signals if batch)
        return self.compute_scms()
    
    def shuffle_scenario(self, idxStart=0, idxEnd=-1):
        """Shuffle the last existing acoustic scenario by applying small
        deviations to source/node positions."""
        c = self.cfg
        asc = self.scenarios[-1]  # use last scenario

        # Apply small deviations to the nodes and sources positions
        def _wiggle_array(arr: np.ndarray):
            wiggles = np.random.uniform(
                -c.dynamicWiggleSize, c.dynamicWiggleSize, size=arr.shape
            )
            return arr + wiggles, wiggles
        
        def _wiggle(currAsc: StaticScenarioParameters):
            ascOut = copy.deepcopy(currAsc)
            ascOut.nodesPos, wiggles = _wiggle_array(asc.nodesPos)
            # Apply node wiggles to sensor positions
            ascOut.sensorsPos = asc.sensorsPos.copy()
            for k in range(c.K):
                ascOut.sensorsPos[c.Mkc[k]:c.Mkc[k + 1], :] += wiggles[k, :]
            ascOut.noiseSourcesPos = _wiggle_array(asc.noiseSourcesPos)[0]
            ascOut.speechSourcesPos = _wiggle_array(asc.speechSourcesPos)[0]
            return ascOut
        
        def _valid_asc(testAsc: StaticScenarioParameters):
            # Check if the minimum distance from any node to any source is respected
            rd = [c.roomLength, c.roomWidth, c.roomHeight]
            allSources = np.vstack([testAsc.speechSourcesPos, testAsc.noiseSourcesPos])
            for k in range(c.K):
                for srcPos in allSources:
                    if np.linalg.norm(testAsc.nodesPos[k, :] - srcPos) < c.minDistNodeSource:
                        return False
            # Check if the minimum distance from any object to the walls is respected
            if np.any([rd - testAsc.sensorsPos < c.minDistFromWall]) or np.any([testAsc.sensorsPos < c.minDistFromWall]):
                return False
            if np.any([rd - allSources < c.minDistFromWall]) or np.any([allSources < c.minDistFromWall]):
                return False
            return True

        wiggledAsc = _wiggle(asc)
        idxTrial = 0
        while not _valid_asc(wiggledAsc):
            print(f'Trial {idxTrial+1}: Invalid scenario, retrying...', end='\r')
            wiggledAsc = _wiggle(asc)
            idxTrial += 1
        print(f'\nWiggled scenario created in {idxTrial+1} trials.')

        # Make a new room
        rd = [c.roomLength, c.roomWidth, c.roomHeight]
        if c.t60 == 0:
            maxOrd = 0
            eAbs = 0.5  # <-- arbitrary
        else:
            eAbs, maxOrd = pra.inverse_sabine(c.t60, rd)
        newRoom = pra.ShoeBox(
            rd,
            fs=c.fs,
            max_order=maxOrd,
            air_absorption=True if c.t60 > 0 else False,
            materials=pra.Material(eAbs),
            use_rand_ism=False
        )

        # Add elements to room again and simulate new RIRs
        rirs = self.simulate_room(
            newRoom,
            wiggledAsc.sensorsPos,
            wiggledAsc.speechSourcesPos,
            wiggledAsc.noiseSourcesPos
        )
        p = StaticScenarioParameters(
            nodesPos=wiggledAsc.nodesPos,
            sensorsPos=wiggledAsc.sensorsPos,
            speechSourcesPos=wiggledAsc.speechSourcesPos,
            noiseSourcesPos=wiggledAsc.noiseSourcesPos,
            obsMat=asc.obsMat,
            Qkq=asc.Qkq,
            rirs=rirs,
            aMat=asc.aMat,
            trees=asc.trees,
        )
        self.apply_static_scenario(p, smIdx=[idxStart, idxEnd])
        # Store the parameters for later use
        self.scenarios.append(p)

    def setup_wola_domain_static(self, idxStart=0, idxEnd=-1, omFixed=None):
        """Setup a static acoustic scenario in the WOLA domain."""
        c = self.cfg
        # Setup the room
        room = self.setup_room()
        # Generate the WASN parameters
        p: StaticScenarioParameters = self.define_static_scenario(room, omFixed=omFixed)
        self.apply_static_scenario(p, smIdx=[idxStart, idxEnd])
        # Store the parameters for later use
        self.scenarios.append(p)
        return room

    def setup_room(self):
        """Setup the room for the acoustic scenario."""
        c = self.cfg
        if not c.customScenarioPartition:
            print(f"Setting up a single-room scenario with T60 = {c.t60} s.")
            # Compute material absorption coefficients from T60
            # using the Sabine formula
            rd = [c.roomLength, c.roomWidth, c.roomHeight]
            if c.t60 == 0:
                maxOrd = 0
                eAbs = 0.5  # <-- arbitrary
            else:
                eAbs, maxOrd = pra.inverse_sabine(c.t60, rd)

            room = pra.ShoeBox(
                rd,
                fs=c.fs,
                max_order=maxOrd,
                air_absorption=True if c.t60 > 0 else False,
                materials=pra.Material(eAbs),
                use_rand_ism=False
            )
        else:
            # Internal partition in top view: vertical segment at x = x_part, from y=c.roomLength down to y=y_bot
            x_part = c.roomWidth / 2
            y_top  = c.roomLength
            y_bot  = c.roomLength / 2
            # Reflection -> absorption
            alpha_external = 1.0 - c.externalWallReflectionCoeff**2
            alpha_internal = 1.0 - c.internalWallReflectionCoeff**2
            # Define materials
            mat_external = pra.Material(energy_absorption=alpha_external, scattering=0.0)
            mat_internal = pra.Material(energy_absorption=alpha_internal, scattering=0.0)

            # -----------------------------
            # Create outer 3D room
            # -----------------------------
            room = pra.ShoeBox(
                [c.roomWidth, c.roomLength, c.roomHeight],
                fs=c.fs,
                materials=mat_external,   # applies to all 6 boundary surfaces
                max_order=10
            )
            # -----------------------------
            # Add internal partition wall (floor-to-ceiling rectangle)
            # -----------------------------
            # Helper: convert Material coeffs to the (m,1) float32 arrays expected by libroom.Wall
            def wall_coeffs_from_material(mat: pra.Material):
                a = np.asarray(mat.absorption_coeffs, dtype=np.float32).reshape(-1, 1)
                # scattering may be None in some cases; default to zeros with matching shape
                if getattr(mat, "scattering_coeffs", None) is None:
                    s = np.zeros_like(a, dtype=np.float32)
                else:
                    s = np.asarray(mat.scattering_coeffs, dtype=np.float32).reshape(-1, 1)
                return a, s

            abs_red, sca_red = wall_coeffs_from_material(mat_internal)

            # Internal partition as a 3D rectangle (3 x 4), floor-to-ceiling
            partition = np.array([
                [x_part, x_part, x_part, x_part],
                [y_top,  y_bot,  y_bot,  y_top ],
                [0.0,    0.0, c.roomHeight, c.roomHeight],
            ], dtype=np.float32)

            # Append wall (no add_wall() method)
            room.walls.append(
                pra.Wall(partition, absorption=abs_red, scattering=sca_red, name="partition")
            )

        return room

    def compute_scms(self, force=None):
        """Compute the SCMs for the acoustic scenario."""
        c = self.cfg

        # Compute centralized signals STFT
        stack = dict()
        for st in self.nodes[0].td.keys():
            if c.wolaMixtures_viaTD:
                if self.nodes[0].td[st].ndim == 3:
                    # Individual source contributions
                    tmp = []
                    for ii in range(self.nodes[0].td[st].shape[0]):
                        tmp.append(
                            np.vstack([self.nodes[k].td[st][ii, ...] for k in range(c.K)])
                        )
                    tmp = np.array(tmp)
                else:
                    tmp = np.vstack([self.nodes[k].td[st] for k in range(c.K)])
                stack[st] = c.get_stft(tmp)
            else:
                stack[st] = np.vstack([self.nodes[k].wd[st] for k in range(c.K)])
        print(f'{c.T} s of signals = {stack["y"].shape[-1]} STFT frames.')

        if force is None:
            scmEstType = c.scmEstimation
        else:
            scmEstType = force

        if scmEstType == 'batch':
            Rnn = np.einsum('ijk,ljk->jil', stack['n'], stack['n'].conj())
            Rss = np.einsum('ijk,ljk->jil', stack['s'], stack['s'].conj())
            if c.noCrossCorrelation:
                Ryy = Rss + Rnn
            else:
                Ryy = np.einsum('ijk,ljk->jil', stack['y'], stack['y'].conj())
        elif scmEstType == 'oracle':
            # Compute FFT of RIRs
            Cmat = np.zeros((c.nPosFreqs, c.M, c.Q), dtype=complex)
            scn = self.scenarios[0]  # Use the first scenario for oracle estimation
            for ii in range(c.Q):
                rirs = np.array([scn.rirs[m][ii] for m in range(c.M)])
                tmp = np.fft.rfft(rirs, n=c.nfft, axis=-1)   # RIRs FFT => transfer functions
                if c.singleLine is not None:
                    # If singleLine is set, only use the specified frequency line
                    tmp = tmp[:, [c.singleLine]]
                # Set the steering vectors of nodes that do not observe source ii to zero
                for q in np.where(scn.obsMat[:, ii] == 0)[0]:
                    tmp[c.Mkc[q]:c.Mkc[q + 1], :] = 0
                Cmat[..., ii] = tmp.T
            # Compute STFTs of latent signals
            slatSTFT = c.get_stft(self.latentDesired)
            nlatSTFT = c.get_stft(self.latentNoise)
            power_s = np.mean(np.abs(slatSTFT) ** 2, axis=-1)
            power_n = np.mean(np.abs(nlatSTFT) ** 2, axis=-1)
            power_v = np.mean(np.abs(stack['sn']) ** 2, axis=-1)
            # Compute the SCMs
            Rss = np.zeros((c.nPosFreqs, c.M, c.M), dtype=complex)
            Rnn = np.zeros((c.nPosFreqs, c.M, c.M), dtype=complex)
            for f in range(c.nPosFreqs):
                Rsslat_f = np.diag(power_s[:, f])
                Rnnlat_f = np.diag(power_n[:, f])
                Rss[f, ...] = Cmat[f, :, :c.Qd] @ Rsslat_f @ Cmat[f, :, :c.Qd].conj().T
                Rnn[f, ...] = Cmat[f, :, c.Qd:] @ Rnnlat_f @ Cmat[f, :, c.Qd:].conj().T
                # Add self-noise to noise SCM
                Rnn[f, ...] += np.diag(power_v[:, f])
            # Complete signal SCM
            Ryy = Rss + Rnn
            
        elif scmEstType == 'online':
            Ryy, Rss, Rnn = None, None, None # nothing to do -- done in `algos.py`
        else:
            raise ValueError(f"Unknown SCM estimation method: {scmEstType}")

        return Ryy, Rss, Rnn, stack

    def compute_vad(self, signal):
        c = self.cfg
        smoothingLength = int(np.amin((c.vadSmoothingPeriod, c.noiseONOFFperiod)) * c.fs)
        # Compute the energy of the signal
        energy = np.mean(np.abs(signal) ** 2)
        # Compute the threshold for VAD
        threshold = c.vadThreshold * energy
        # Smooth the VAD: avoid abrupt changes between 1 and 0
        return np.convolve(
            np.abs(signal) ** 2 > threshold,
            np.ones(smoothingLength) / smoothingLength,
            mode='same'
        ) > 0.5

    def estimate_vad(self):
        """Estimate the VAD for the latent desired signal."""
        c = self.cfg
        vad = np.zeros((c.Qd, c.N), dtype=bool)
        for ii in range(c.Qd):
            vad[ii, :] = self.compute_vad(self.latentDesired[ii, :])

        # Convert to VAD in STFT domain: VAD in one frame is True if at least
        # half of the samples are True.
        vadSTFT = np.zeros((c.Qd, c.nFrames), dtype=bool)
        for l in range(c.nFrames):
            idxBeg = int(l * (c.nfft - c.nhop))
            idxEnd = idxBeg + c.nfft
            for ii in range(c.Qd):
                vadSTFT[ii, l] = np.sum(vad[ii, idxBeg:idxEnd]) > c.nfft / 2

        return vad.astype(bool), vadSTFT.astype(bool)

    def plot(self, scenarioIdx=0):
        """Export the environment to a TXT file and to a plot."""
        c = self.cfg
        scn = self.scenarios[scenarioIdx]  # Use the first scenario for plotting
        # Make a plot of the environment
        if c.onPlane:
            figObs, axes = plt.subplots(1, 1)
            figObs.set_size_inches(5.5, 5.5)
            # Plot the room
            axes.plot([0, c.roomLength], [0, 0], 'k-')
            axes.plot([0, 0], [0, c.roomWidth], 'k-')
            axes.plot([c.roomLength, c.roomLength], [0, c.roomWidth], 'k-')
            axes.plot([0, c.roomLength], [c.roomWidth, c.roomWidth], 'k-')
            # Plot the nodes
            for k in range(c.K):
                # axes.plot(scn.nodesPos[k, 0], scn.nodesPos[k, 1], 'ro', markersize=2)
                axes.text(scn.nodesPos[k, 0] + c.nodeRadius, scn.nodesPos[k, 1] + c.nodeRadius, str(k+1))
            # Plot the sensors
            for k in range(c.K):
                for m in range(c.Mk[k]):
                    axes.plot(
                        scn.sensorsPos[c.Mkc[k] + m, 0],
                        scn.sensorsPos[c.Mkc[k] + m, 1],
                        'ko', markersize=2
                    )
            # Add circle around the nodes 
            for k in range(c.K):
                circle = plt.Circle(
                    (scn.nodesPos[k, 0], scn.nodesPos[k, 1]),
                    c.nodeRadius * 1.2,
                    color='k',
                    fill=True,
                    alpha=0.2,
                    edgecolor='k',
                    linestyle='--'
                )
                axes.add_artist(circle)
            # Plot the sources
            for ii in range(c.Qd):
                axes.plot(
                    scn.speechSourcesPos[ii, 0],
                    scn.speechSourcesPos[ii, 1], 'go')
                axes.text(
                    scn.speechSourcesPos[ii, 0],
                    scn.speechSourcesPos[ii, 1], f'S{ii+1}')
            for ii in range(c.Qn):
                axes.plot(
                    scn.noiseSourcesPos[ii, 0],
                    scn.noiseSourcesPos[ii, 1], 'rd')
                axes.text(
                    scn.noiseSourcesPos[ii, 0],
                    scn.noiseSourcesPos[ii, 1], f'N{ii+1}')
            # Plot the observability matrix
            axes.grid()
            axes.set_aspect('equal')
            axes.set_title(f"Acoustic environment (scenario {scenarioIdx+1}/{len(self.scenarios)})")
            figObs.tight_layout()

            # Create a copy of the figure without observability lines
            fig = copy.deepcopy(figObs)

            # Add lines between nodes and sources to indicate observability
            for k in range(c.K):
                for ii in range(c.Qd + c.Qn):
                    if scn.obsMat[k, ii] == 1:
                        if ii < c.Qd:
                            axes.plot(
                                [scn.nodesPos[k, 0], scn.speechSourcesPos[ii, 0]],
                                [scn.nodesPos[k, 1], scn.speechSourcesPos[ii, 1]],
                                'g:',
                                alpha=0.5
                            )
                        else:
                            axes.plot(
                                [scn.nodesPos[k, 0], scn.noiseSourcesPos[ii - c.Qd, 0]],
                                [scn.nodesPos[k, 1], scn.noiseSourcesPos[ii - c.Qd, 1]],
                                'r:',
                                alpha=0.5
                            )
            axes.set_title(f"Observabilities")
        else:
            # Plot in a 3D space
            figObs = plt.figure()
            figObs.set_size_inches(5.5, 5.5)
            ax = figObs.add_subplot(111, projection='3d')
            # Plot the room
            ax.plot([0, c.roomLength], [0, 0], [0, 0], 'k-', alpha=0.5)
            ax.plot([0, 0], [0, c.roomWidth], [0, 0], 'k-', alpha=0.5)
            ax.plot([c.roomLength, c.roomLength], [0, c.roomWidth], [0, 0], 'k-', alpha=0.5)
            ax.plot([0, c.roomLength], [c.roomWidth, c.roomWidth], [0, 0], 'k-', alpha=0.5)
            ax.plot([0, 0], [0, 0], [0, c.roomHeight], 'k-', alpha=0.5)
            ax.plot([0, 0], [c.roomWidth, c.roomWidth], [0, c.roomHeight], 'k-', alpha=0.5)
            ax.plot([c.roomLength, c.roomLength], [0, 0], [0, c.roomHeight], 'k-', alpha=0.5)
            ax.plot([c.roomLength, c.roomLength], [c.roomWidth, c.roomWidth], [0, c.roomHeight], 'k-', alpha=0.5)
            ax.plot([0, c.roomLength], [0, 0], [c.roomHeight, c.roomHeight], 'k-', alpha=0.5)
            ax.plot([0, 0], [0, c.roomWidth], [c.roomHeight, c.roomHeight], 'k-', alpha=0.5)
            ax.plot([c.roomLength, c.roomLength], [0, 0], [c.roomHeight, c.roomHeight], 'k-', alpha=0.5)
            ax.plot([0, c.roomLength], [c.roomWidth, c.roomWidth], [c.roomHeight, c.roomHeight], 'k-', alpha=0.5)
            ax.plot([c.roomLength, c.roomLength], [0, c.roomWidth], [c.roomHeight, c.roomHeight], 'k-', alpha=0.5)
            # Plot the nodes
            for k in range(c.K):
                ax.plot(
                    scn.nodesPos[k, 0], scn.nodesPos[k, 1], scn.nodesPos[k, 2],
                    'bo', markersize=2 * 50 * c.nodeRadius, alpha=0.5
                )
                ax.text(scn.nodesPos[k, 0] + c.nodeRadius, scn.nodesPos[k, 1] + c.nodeRadius, scn.nodesPos[k, 2] + c.nodeRadius, str(k+1))
                # Add vertical line from node to the floor
                ax.plot(
                    [scn.nodesPos[k, 0], scn.nodesPos[k, 0]],
                    [scn.nodesPos[k, 1], scn.nodesPos[k, 1]],
                    [0, scn.nodesPos[k, 2]], 'k--', alpha=0.5
                )
            # Plot the sensors
            for k in range(c.K):
                for m in range(c.Mk[k]):
                    ax.plot(
                        scn.sensorsPos[c.Mkc[k] + m, 0],
                        scn.sensorsPos[c.Mkc[k] + m, 1],
                        scn.sensorsPos[c.Mkc[k] + m, 2],
                        'ko', markersize=2
                    )
            # Plot the sources
            for ii in range(c.Qd):
                ax.plot(
                    scn.speechSourcesPos[ii, 0],
                    scn.speechSourcesPos[ii, 1],
                    scn.speechSourcesPos[ii, 2], 'go')
                ax.text(
                    scn.speechSourcesPos[ii, 0],
                    scn.speechSourcesPos[ii, 1],
                    scn.speechSourcesPos[ii, 2], f'S{ii+1}')
            for ii in range(c.Qn):
                ax.plot(
                    scn.noiseSourcesPos[ii, 0],
                    scn.noiseSourcesPos[ii, 1],
                    scn.noiseSourcesPos[ii, 2], 'rd')
                ax.text(
                    scn.noiseSourcesPos[ii, 0],
                    scn.noiseSourcesPos[ii, 1],
                    scn.noiseSourcesPos[ii, 2], f'N{ii+1}')
            # Add parition (if any), as transparent area
            if c.customScenarioPartition:
                x_part = c.roomWidth / 2
                y_top  = c.roomLength
                y_bot  = c.roomLength / 2
                z_bot  = 0.0
                z_top  = c.roomHeight
                xx, zz = np.meshgrid(
                    np.array([x_part, x_part]),
                    np.array([z_bot, z_top])
                )
                yy = np.array([[y_bot, y_bot], [y_top, y_top]])
                ax.plot_surface(
                    xx, yy, zz.T,
                    color='gray',
                    alpha=0.3,
                    edgecolor='k'
                )
            ax.set_xlabel('Length [m]')
            ax.set_ylabel('Width [m]')
            ax.set_zlabel('Height [m]')
            ax.set_title(f"Acoustic environment (scenario {scenarioIdx+1}/{len(self.scenarios)})")
        return fig, figObs

    def plot_dynamic(self):
        # Overlay several scenarios
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for scenarioIdx, scn in enumerate(self.scenarios):
            ax.scatter(scn.nodesPos[:, 0], scn.nodesPos[:, 1], scn.nodesPos[:, 2], label=f'Scenario {scenarioIdx+1}')
        ax.set_xlabel('Length [m]')
        ax.set_ylabel('Width [m]')
        ax.set_zlabel('Height [m]')
        ax.set_title('Acoustic environment (dynamic)')
        ax.legend()
        plt.show()

    def store_scenario_parameters(self, p: StaticScenarioParameters):
        """Store the scenario parameters in the environment."""
        self.nodesPos = p.nodesPos
        self.sensorsPos = p.sensorsPos
        self.speechSourcesPos = p.speechSourcesPos
        self.noiseSourcesPos = p.noiseSourcesPos
        self.obsMat = p.obsMat
        self.aMat = p.aMat
        self.Qkq = p.Qkq
        self.rirs = p.rirs
        if p.aMat is not None:
            self.trees = p.trees

    def gen_latent_desired(self, n: int) -> list[str]:
        """Generate speech signals using files from the database."""
        c = self.cfg
        if c.desSigType == 'random':
            # Generate white noise signals
            randsigs = c.randmat((n, c.N), makeComplex=False)
            if c.useVAD:
                # Add pauses to the random signals
                pauseLength = int(c.fs * c.noiseONOFFperiod)
                for ii in range(n):
                    # Randomly select a start index for the pause
                    start = np.random.randint(0, pauseLength // 4)
                    start = 0  # DEBUG !!! TODO: remove this line
                    starts = np.arange(start, c.N - pauseLength + 1, 2 * pauseLength)
                    for s in starts:
                        randsigs[ii, s:s + pauseLength] = 0
            return randsigs
        elif c.desSigType == 'speech':
            if 'VCTK' in c.speechDatabasePath:
                d = SPEECH_DATABASE_NAME_DELIMITER['VCTK']
            else:
                raise ValueError("Unknown speech database. Please specify the delimiter.")
            # Recursively search for all .wav files in the database path
            path = Path(c.speechDatabasePath)
            files = list(path.rglob("*.wav")) + list(path.rglob("*.flac"))
            if len(files) == 0:
                raise ValueError("No files found in the database path.")
            nSamples = c.N
            x = np.zeros((n, nSamples), dtype='float32')

            def _randsel(myfiles):
                fr = myfiles[np.random.randint(0, len(myfiles), 1)[0]]
                while fr.name[:d] in alreadyUsed:
                    fr = myfiles[np.random.randint(0, len(myfiles), 1)[0]]
                alreadyUsed.append(fr.name[:d])
                if '-' in fr.name:
                    prefix = fr.name.split("-")[0]
                elif '_' in fr.name:
                    prefix = fr.name.split("_")[0]
                return fr, prefix, alreadyUsed

            alreadyUsed = []
            for ii in range(n):
                print(f'Building latent speech signal {ii+1}/{n}...')
                # Randomly select one file from the list
                fr, prefix, alreadyUsed = _randsel(files)
                # Load file 
                speech = self.load_sound_file(fr)
                idx = 0
                while len(speech) < nSamples:
                    print(f'Loading and processing speech snippet #{idx+1}...', end='\r')
                    # Find another file with the same prefix
                    filesSamePrefix = [f for f in files if f.name.startswith(prefix) and f.name not in alreadyUsed]
                    if len(filesSamePrefix) == 0:
                        print(f"No more files found with the same prefix as {fr}. Changing prefix at {len(speech)} samples (= {len(speech)/c.fs:.2f} s)...")
                        # Randomly select one file from the list
                        fr, prefix, alreadyUsed = _randsel(files)
                        filesSamePrefix = [f for f in files if f.name.startswith(prefix) and f.name not in alreadyUsed]
                        idx = 0
                    else:
                        # Randomly select one file from the list
                        fr = filesSamePrefix[np.random.randint(0, len(filesSamePrefix), 1)[0]]
                        while fr.name[:d] in alreadyUsed:
                            fr = filesSamePrefix[np.random.randint(0, len(filesSamePrefix), 1)[0]]
                    alreadyUsed.append(fr.name[:d])
                    # Concat the file to the speech signal
                    newSig = self.load_sound_file(fr)
                    speech = np.concatenate((speech, newSig))
                    idx += 1
                x[ii, :] = speech[:nSamples]
            print(f"\nBuilt latent speech signal.")
            return x
        
    def get_latent_noise(self, n: int) -> np.ndarray:
        """Generate noise signals."""
        c = self.cfg
        if c.noiseSigType == 'random':
            # Generate white noise signals
            noise = c.randmat((n, c.N), makeComplex=False)
            noise /= np.amax(np.abs(noise))  # Normalize
            noise -= np.mean(noise)  # Remove DC offset
            return noise
        elif c.noiseSigType in ['ssn', 'babble']:
            out = np.zeros((n, c.N), dtype='float32')

            # Select noise files from the database
            if c.noiseSigType == 'ssn':
                allFiles = list(Path(c.ssnDatabasePath).rglob("ssn*.wav"))
            elif c.noiseSigType == 'babble':
                allFiles = list(Path(c.babbleDatabasePath).rglob("babble*.wav"))

            for ii in range(n):
                # Randomly select one file from the SSN database
                if ii >= len(allFiles):
                    raise ValueError(f"Not enough files in the database (asked for {n}, found {len(allFiles)}).")
                else:
                    currFile = allFiles[ii]
                tmp = self.load_sound_file(currFile, noise=True)
                while len(tmp) < c.N:
                    # If the file is too short, concatenate it with another file
                    tmp2 = self.load_sound_file(currFile, noise=True)
                    tmp = np.concatenate((tmp, tmp2))
                out[ii, :] = tmp[:c.N]
            return out

    def apply_static_scenario(self, p: StaticScenarioParameters, smIdx: list[int] = None):
        c = self.cfg  # Configuration object

        if smIdx is None:
            smIdx = np.array([0, c.N - 1], dtype=int)  # Default indices for the whole signal
        if smIdx[-1] == -1:
            smIdx[-1] = c.N

        if c.wolaMixtures_viaTD:
            # Apply RIRs via fftconvolve, returning time domain signals
            self.apply_static_scenario_viaTD(p, smIdx)
        else:
            # Apply FFT of RIRs in the STFT domain, returning STFT signals
            self.apply_static_scenario_viaSTFT(p, smIdx)

    def apply_static_scenario_viaTD(self, p: StaticScenarioParameters, smIdx: list[int]):
        """
        Apply the static scenario in the time domain using room impulse responses.
        This method applies the RIRs to the latent signals via fftconvolve
        and computes the time domain signals for each node, taking the 
        observability matrix into account.
        
        Parameters
        ----------
        p : StaticScenarioParameters
            The static scenario parameters.
        smIdx : list[int], optional
            The indices of the signal segments to process (default is the whole signal).
        """
        c = self.cfg  # Configuration object
        # Prepare potential window smoothing
        if c.dynamics == 'moving':
            smIdxEff = [
                np.amax((0, smIdx[0] - c.dynTransLen)),
                np.amin((c.N, smIdx[1] + c.dynTransLen))
            ]
            smoothInLen = smIdx[0] - smIdxEff[0]
            smoothOutLen = smIdxEff[1] - smIdx[1]
            # Prepare smoothing windows
            smoothInWin = sig.get_window(c.win, 2 * smoothInLen)
            smoothIn = smoothInWin[:smoothInLen]
            smoothOutWin = sig.get_window(c.win, 2 * smoothOutLen)
            smoothOut = smoothOutWin[smoothOutLen:]
            fullSmoothWin = np.concatenate((smoothIn, np.ones(smIdx[1] - smIdx[0]), smoothOut))
        else:
            smIdxEff = smIdx
            fullSmoothWin = np.ones(smIdx[1] - smIdx[0])
        # Apply the room impulse responses to the latent signals
        for k in range(c.K):
            # Get the indices of the microphones for this node
            micIdx = np.arange(c.Mkc[k], c.Mkc[k + 1])
            # Apply the room impulse responses to the latent signals
            for ii in range(c.Qd):
                # ---------------------------------------
                if p.obsMat[k, ii] == 0 and not c.customScenarioPartition:
                    continue  # Skip if the node does not observe the source
                # ---------------------------------------
                for jj, m in enumerate(micIdx):
                    tmp = sig.fftconvolve(
                        self.latentDesired[ii, smIdxEff[0]:smIdxEff[1]],
                        p.rirs[m][ii],
                    )[:-(c.nfft - 1)]
                    tmp = tmp[:len(fullSmoothWin)] * fullSmoothWin
                    self.nodes[k].td['sIndiv'][ii, jj, smIdxEff[0]:smIdxEff[1]] += tmp
            for ii in range(c.Qn):
                # ---------------------------------------
                if p.obsMat[k, c.Qd + ii] == 0 and not c.customScenarioPartition:
                    continue  # Skip if the node does not observe the source
                # ---------------------------------------
                for jj, m in enumerate(micIdx):
                    tmp = sig.fftconvolve(
                        self.latentNoise[ii, smIdxEff[0]:smIdxEff[1]],
                        p.rirs[m][c.Qd + ii]
                    )[:-(c.nfft - 1)]
                    tmp = tmp[:len(fullSmoothWin)] * fullSmoothWin
                    self.nodes[k].td['nIndiv'][ii, jj, smIdxEff[0]:smIdxEff[1]] += tmp
            self.nodes[k].td['s'] = np.sum(self.nodes[k].td['sIndiv'], axis=0)
            self.nodes[k].td['n'] = np.sum(self.nodes[k].td['nIndiv'], axis=0)

    def apply_static_scenario_viaSTFT(self, p: StaticScenarioParameters, smIdx: list[int]):
        """
        Apply the static scenario in the STFT domain using room impulse responses.
        This method applies the FFT of RIRs to the STFT of latent signals, taking
        the observability matrix into account, and computes the resulting STFT
        signals for each node.

        Parameters
        ----------
        p : StaticScenarioParameters
            The static scenario parameters.
        smIdx : list[int], optional
            The indices of the signal segments to process (default is the whole signal).
            This is used to determine the frames to process.
        """
        c = self.cfg
        # Cmat = [A | B], where A is the steering matrix for desired sources
        # and B is the steering matrix for noise sources.
        Cmat = np.zeros((c.nPosFreqs, c.M, c.Q), dtype=complex)
        for ii in range(c.Q):
            rirs = np.array([p.rirs[m][ii] for m in range(c.M)])
            tmp = np.fft.rfft(rirs, n=c.nfft, axis=-1)   # RIRs FFT (= transfer functions)
            if c.singleLine is not None:
                # Only use one frequency line
                tmp = tmp[:, [c.singleLine]]
            # Set the steering vectors of nodes that do not observe source ii to zero
            if not c.customScenarioPartition:
                for q in np.where(p.obsMat[:, ii] == 0)[0]:
                    tmp[c.Mkc[q]:c.Mkc[q + 1], :] = 0
            Cmat[..., ii] = tmp.T
        slatSTFT = c.get_stft(self.latentDesired)
        nlatSTFT = c.get_stft(self.latentNoise)
        idxFrameBeg = int(np.ceil(smIdx[0] / (c.nfft - c.nhop)))
        idxFrameEnd = int(np.ceil(smIdx[1] / (c.nfft - c.nhop))) + 1
        s = np.einsum(
            'ijk,kil->jil',
            Cmat[..., :c.Qd],  # Desired sources steering matrix
            slatSTFT[..., idxFrameBeg:idxFrameEnd]
        )
        n = np.einsum(
            'ijk,kil->jil',
            Cmat[..., c.Qd:],  # Noise sources steering matrix
            nlatSTFT[..., idxFrameBeg:idxFrameEnd]
        )

        pass

        # Store the STFT signals in the nodes
        for k in range(c.K):
            self.nodes[k].wd['s'][..., idxFrameBeg:idxFrameEnd] = s[c.Mkc[k]:c.Mkc[k + 1], ...]
            self.nodes[k].wd['n'][..., idxFrameBeg:idxFrameEnd] = n[c.Mkc[k]:c.Mkc[k + 1], ...]

    def define_layout(self, walls: list[pra.libroom.Wall]=None):
        """Define the layout of the acoustic scenario."""
        c = self.cfg
        rd = [c.roomLength, c.roomWidth, c.roomHeight]
        zPlane = 1.3 if c.roomHeight > 1.3 else c.roomHeight / 2.0

        if c.unconstrainedRandomPositions:
            nodePos = np.random.rand(c.K, 3) * np.array(rd)
            sensorsPos = np.random.rand(c.M, 3) * np.array(rd)
            speechSourcesPos = np.random.rand(c.Qd, 3) * np.array(rd)
            noiseSourcesPos = np.random.rand(c.Qn, 3) * np.array(rd)
            if c.onPlane:
                # If the scenario is on a plane, set the z-coordinate to a fixed value
                nodePos[:, 2] = zPlane
                sensorsPos[:, 2] = zPlane
                speechSourcesPos[:, 2] = zPlane
                noiseSourcesPos[:, 2] = zPlane
            return nodePos, sensorsPos, speechSourcesPos, noiseSourcesPos

        # Generate node positions
        nodesPos = np.zeros((c.K, 3))
        for k in range(c.K):
            tmp = sample_position_from_room(
                walls=walls,
                min_dist=c.minDistFromWall,
            )
            if c.onPlane:
                tmp[2] = zPlane
            while np.any(
                np.linalg.norm(tmp - np.array(nodesPos[:k, :]), axis=1) < c.nodeRadius * 2.5
            ):
                print(f"Node {k + 1}/{c.K} too close to another node, generating a new position...", end='\r')
                # If the node is too close to another node, generate a new position
                tmp = sample_position_from_room(
                    walls=walls,
                    min_dist=c.minDistFromWall,
                )
                if c.onPlane:
                    tmp[2] = zPlane
            # Store the position of the node
            nodesPos[k, :] = tmp
        
        # Generate sensor positions around node position
        # sensorsPos = np.zeros((c.K, c.Mk, 3))
        sensorsPos = [np.zeros((c.Mk[k], 3)) for k in range(c.K)]
        for k in range(c.K):
            sensors = np.vstack(
                [nodesPos[k, :]] * c.Mk[k]
            )  # Start with the node position as the first sensor
            for m in range(c.Mk[k]):
                # Generate a random position around the node position
                tmp = nodesPos[k, :] + np.random.randn(3) * (c.nodeRadius / 2)
                if c.onPlane:
                    tmp[2] = zPlane
                while np.any(np.linalg.norm(tmp - np.array(sensors), axis=1) < c.nodeRadius / c.Mk[k]):
                    print(f"Sensor {m + 1}/{c.Mk[k]} of node {k + 1}/{c.K} too close to another sensor, generating a new position...", end='\r')
                    # If the sensor is too close to another sensor, generate a new position
                    tmp = nodesPos[k, :] + np.random.randn(3) * (c.nodeRadius / 2)
                    if c.onPlane:
                        tmp[2] = zPlane
                # Store the position of the sensor
                sensors[m, :] = tmp
            sensorsPos[k] = np.array(sensors)
        # Flatten the sensor positions
        sensorsPos = np.vstack(sensorsPos)

        if c.onPlane:
            # Ensure that the nodes are on the same plane
            zPlane = 1.3 if c.roomHeight > 1.3 else c.roomHeight / 2.0
            nodesPos[:, 2] = zPlane
            sensorsPos[:, 2] = zPlane

        # Generate source positions
        speechSourcesPos = np.zeros((c.Qd, 3))
        for ii in range(c.Qd):
            attempt = sample_position_from_room(
                walls=walls,
                min_dist=c.minDistFromWall,
            )
            if c.onPlane:
                attempt[2] = zPlane
            counter = 0
            while np.linalg.norm(attempt - nodesPos, axis=1).min() < c.minDistNodeSource:
                print(f"Desired source {ii + 1}/{c.Qd} too close to a node, generating a new position (trial #{counter+1})...", end='\r')
                # If the source is too close to a node, generate a new position
                attempt = sample_position_from_room(
                    walls=walls,
                    min_dist=c.minDistFromWall,
                )
                if c.onPlane:
                    attempt[2] = zPlane
                counter += 1
            # Store the position of the source
            speechSourcesPos[ii, :] = attempt

        # Generate noise source positions
        noiseSourcesPos = np.zeros((c.Qn, 3))
        for ii in range(c.Qn):
            attempt = sample_position_from_room(
                walls=walls,
                min_dist=c.minDistFromWall,
            )
            if c.onPlane:
                attempt[2] = zPlane
            counter = 0
            while np.linalg.norm(attempt - nodesPos, axis=1).min() < c.minDistNodeSource or\
                np.linalg.norm(attempt - speechSourcesPos, axis=1).min() < c.minDistNodeSource:
                print(f"Noise source {ii + 1}/{c.Qn} too close to a node or a desired source, generating a new position (trial #{counter+1})...", end='\r')
                # If the source is too close to a node, generate a new position
                attempt = sample_position_from_room(
                    walls=walls,
                    min_dist=c.minDistFromWall,
                )
                if c.onPlane:
                    attempt[2] = zPlane
                counter += 1
            # Store the position of the source
            noiseSourcesPos[ii, :] = attempt

        return nodesPos, sensorsPos, speechSourcesPos, noiseSourcesPos

    def simulate_room(
            self,
            room: pra.ShoeBox,
            sensorsPos: np.ndarray,
            speechSourcesPos: np.ndarray,
            noiseSourcesPos: np.ndarray
        ):
        c = self.cfg
        # Add sources to the room
        for ii in range(c.Qd):
            room.add_source(speechSourcesPos[ii, :])
        for ii in range(c.Qn):
            room.add_source(noiseSourcesPos[ii, :])
        # Add sensors to the room
        room.add_microphone_array(pra.MicrophoneArray(sensorsPos.T, c.fs))
        # Simulate the room
        print("\nComputing the room impulse responses...")
        t0 = time.time()
        room.compute_rir()
        print("Room impulse responses computed in {:.2f} s".format(time.time() - t0))

        if c.truncateRIRsNarrowbandAssumption:
            # Truncate RIRs to ensure respect of narrowband assumption
            for ii in range(len(room.rir)):
                for jj in range(len(room.rir[ii])):
                    # Truncate the RIRs to the first c.nfft samples
                    if len(room.rir[ii][jj]) > c.nfft:
                        room.rir[ii][jj] = room.rir[ii][jj][:c.nfft]
                    elif len(room.rir[ii][jj]) < c.nfft:
                        # Pad the RIRs with zeros to the first c.nfft samples
                        room.rir[ii][jj] = np.pad(
                            room.rir[ii][jj],
                            (0, c.nfft - len(room.rir[ii][jj])),
                            mode='constant'
                        )
        return room.rir

    def define_static_scenario(self, room: pra.ShoeBox, omFixed=None) -> StaticScenarioParameters:
        c = self.cfg  # Configuration object

        # Define the layout of the acoustic scenario
        nodesPos, sensorsPos, speechSourcesPos, noiseSourcesPos =\
            self.define_layout(room.walls)
        
        if omFixed is None:
            # Create observability matrix based on the node and source positions
            obsMat = self.get_observability_matrix(
                nodesPos, speechSourcesPos, noiseSourcesPos
            )  # node x source
        else:
            # Use the fixed observability matrix
            obsMat = omFixed
        
        # Compute number of common sources for each pair of nodes
        Qkq = np.array([[
            np.sum(obsMat[k, :] + obsMat[q, :] == 2)
            for q in range(c.K)
        ] for k in range(c.K)])
        # if np.any(Qkq - np.diag(np.diag(Qkq)) > c.Mk) and any('dMWF' in alg for alg in c.algos):
        #     raise ValueError("Number of common sources exceeds number of sensors per node.")

        # For topology-independent (TI) algorithms, generate the adjacency matrix
        flagTI = any('TI' in alg for alg in c.algos)
        if flagTI:
            # Non-fully connected network
            aMat, commDist = get_adjacency_matrix(c, nodesPos)
            # Get pruned trees and corresponding useful variables, for each
            # possible root node index
            trees = [
                self.tree_pruning(
                    k, sensorsPos,
                    aMat, obsMat,
                    method=c.pruningStrategy
                )
                for k in range(c.K)
            ]

        rirs = self.simulate_room(
            room,
            sensorsPos,
            speechSourcesPos,
            noiseSourcesPos
        )

        return StaticScenarioParameters(
            nodesPos=nodesPos,
            sensorsPos=sensorsPos,
            speechSourcesPos=speechSourcesPos,
            noiseSourcesPos=noiseSourcesPos,
            obsMat=obsMat,
            Qkq=Qkq,
            rirs=rirs,
            aMat=aMat if flagTI else None,
            trees=trees if flagTI else None,
        )

    def get_observability_matrix(
            self,
            nodesPos=None,
            speechPos=None,
            noisePos=None
        ) -> np.ndarray:
        """Compute the observability matrix."""
        c = self.cfg
        if c.observability == 'foss':
            # Full observability -- all nodes observe all sources
            obsMat = np.ones((c.K, c.Qd + c.Qn))
        elif c.observability == 'gls':
            # Global-local subspaces (GLS) scenario
            obsMat = np.zeros((c.K, c.Q))
            def inadequate(om):
                observedDesired = np.sum(om[:, :c.Qd], axis=1) > 0
                observedNoise = np.sum(om[:, c.Qd:], axis=1) > 0
                oneOrAll = (np.sum(om, axis=0) == 1) | (np.sum(om, axis=0) == c.K)
                return not np.all(observedDesired) or\
                    not np.all(observedNoise) or not np.all(oneOrAll)
            # Criterion for adequacy: at least one desired source and one noise
            # source must be observed by each node, and each source must be
            # observed by either one node or all nodes. FOS/FODS scenarios
            # are considered inadequate too, i.e., at least one desired source
            # must be observed by only one node.
            print("Generating observability matrix for GLS scenario...")
            counter = 0
            while inadequate(obsMat):
                print(f"Trial #{counter+1}...", end='\r')
                obsMat = np.random.randint(0, 2, (c.K, c.Q))
                counter += 1
            print(f"\nGenerated observability matrix for GLS scenario in {counter} trials.")
        elif c.observability == 'poss':
            # Partial observability -- nodes only observe certain sources
            # if nodesPos is None or speechPos is None or noisePos is None:
            if 1:  # <--- OVERRIDING: TODO: change that
                # Do not differentiate between global and local sources, 
                # randomly generate observability pattern
                obsMat = np.zeros((c.K, c.Q))
                def inadequate(om):
                    observedDesired = np.sum(om[:, :c.Qd], axis=1) > 0
                    observedNoise = np.sum(om[:, c.Qd:], axis=1) > 0
                    oneObserver = np.sum(om, axis=0) > 0
                    return not np.all(observedDesired) or\
                        not np.all(observedNoise) or not np.all(oneObserver)
                        # not np.all(oneObserver)
                # Criterion for adequacy: at least one desired source and one noise
                # source must be observed by each node, and each source must be
                # observed by at least one node.
                # while inadequate(self.obsMat[:, :c.Qd]) or inadequate(self.obsMat[:, c.Qd:]):
                print("Generating observability matrix for POS scenario...")
                counter = 0
                while inadequate(obsMat):
                    print(f"Trial #{counter+1}...", end='\r')
                    obsMat = np.random.randint(0, 2, (c.K, c.Q))
                    counter += 1
                print(f"\nGenerated observability matrix for POS scenario in {counter} trials.")
            else:
                # Create the distance matrix
                distMat = np.zeros((c.K, c.Qd + c.Qn))
                for k in range(c.K):
                    for d in range(c.Qd):
                        # Compute the distance between the node and the source
                        distMat[k, d] = np.linalg.norm(nodesPos[k, :] - speechPos[d, :])
                    for n in range(c.Qn):
                        # Compute the distance between the node and the noise source
                        distMat[k, c.Qd + n] = np.linalg.norm(nodesPos[k, :] - noisePos[n, :])
                # Threshold the distance matrix to obtain the observability matrix.
                # Adapt the threshold as long as some nodes do not observe any desired
                # source or any noise source, and as long as there exist sources
                # that are not observed by any node.
                thrs = c.maxDistForObservability
                if thrs is None:
                    thrs = np.inf  # No threshold -- full observability
                def inadequacy_criterion(om):
                    return (c.Qd > 0 and np.any(np.sum(om[:, :c.Qd], axis=1) == 0)) |\
                        (c.Qn > 0 and np.any(np.sum(om[:, c.Qd:], axis=1) == 0)) |\
                        np.any(np.sum(om, axis=0) == 0)
                
                # Initialize the observability matrix
                if c.observabilityCriterion == 'raw_distance':
                    obsMat = np.zeros((c.K, c.Qd + c.Qn))
                    while inadequacy_criterion(obsMat):
                        # Obtain observability by thresholding the distance matrix
                        obsMat[distMat <= thrs] = 1
                        if inadequacy_criterion(obsMat):
                            # Increase the threshold
                            thrs *= 1.1
                            if thrs == np.inf:
                                raise ValueError("No threshold can ensure observability for all nodes and sources.")
                            print(f"[Observability matrix] Increasing the threshold to {thrs:.2f} m", end='\r')
                elif c.observabilityCriterion == 'hierarchical':
                    # Start with full observability
                    obsMat = np.ones((c.K, c.Qd + c.Qn))
                    # Remove observability connections starting from the largest
                    # node-source distance, until the inadequacy criterion is met.

                    # Find indices of largest element in the distance matrix
                    counter = 0
                    obsMats = []
                    while not inadequacy_criterion(obsMat):
                        idx = np.unravel_index(np.argmax(distMat, axis=None), distMat.shape)
                        obsMat[idx] = 0  # Remove the observability connection
                        obsMats.append(obsMat.copy())
                        # Prepare the distance matrix for the next iteration
                        distMat[idx] = 0
                        counter += 1
                    # Find the desired amount of observability connections to remove
                    if c.hierarchicalObsPruningThrs > 0:
                        nToRemove = int(c.hierarchicalObsPruningThrs * counter) - 1
                        obsMat = obsMats[nToRemove]
                    else:
                        obsMat = obsMats[0]
        return obsMat  # node x source

    def tree_pruning(
            self,
            root: int,
            sensorsPos: np.ndarray,
            aMat: np.ndarray, 
            obsMat: np.ndarray,
            method: str = 'mst'
        ) -> TreeWASN:
        """Prune the WASN to a tree rooted at node `root`, return corresponding
        adjacency matrix."""
        c = self.cfg
        # Generate NetworkX graph
        Gnx: nx.Graph = nx.from_numpy_array(aMat)
        # Get node positions 
        nodesPos = dict(
            [(k, sensorsPos[c.Mkc[k]:c.Mkc[k + 1], ...]) for k in range(c.K)]
        )
        
        # Add edge weights based on inter-node distance ((TODO -- is that a correct approach?))
        for e in Gnx.edges():
            weight = np.linalg.norm(nodesPos[e[0]] - nodesPos[e[1]])
            Gnx[e[0]][e[1]]['weight'] = weight

        if method == 'mst':
            # ------------ Prune to minimum spanning tree ------------
            # Compute minimum spanning tree
            prunedWasnNX: nx.Graph = nx.minimum_spanning_tree(
                Gnx,
                weight='weight',
                algorithm='kruskal'
            )
        elif method == 'mmut':
            # ------------ Prune to MMUT ------------
            prunedWasnNX = mmut_pruning(Gnx, root=root)
        elif method == 'star':
            # ------------ Prune to star topology ------------
            # Create a star topology with the root node at the center
            iterableNodes = [n for n in Gnx.nodes if n != root]
            iterableNodes.insert(0, root)  # iterable of nodes with root at the center
            prunedWasnNX = nx.star_graph(iterableNodes)
        elif method == 'line':
            # ------------ Prune to line topology ------------
            prunedWasnNX = nx.path_graph(Gnx)
        else:
            raise ValueError(f"Unknown pruning strategy: {method}")
        
        adjMat = nx.adjacency_matrix(prunedWasnNX).toarray()
        adjMat[adjMat > 0] = 1

        # Get the upstream nodes of each node
        upstNodes, upstNeighbors = get_upstream_nodes(prunedWasnNX, root)
        downNodes, downNeighbors = get_downstream_nodes(prunedWasnNX, root)
        # Get the tree structure (list of leaves to root, per level)
        treeStructure = get_tree_structure(prunedWasnNX, root)
        # Get the path from each node to the root
        pathToRoot = [None for _ in range(c.K)]
        for q in range(c.K):
            if q != root:
                pathToRoot[q] = nx.shortest_path(prunedWasnNX, source=q, target=root)

        # Get list of `lq`'s, i.e., children of `u` that are (grand-)parents of `k`
        nq = [None for _ in range(c.K)]
        for q in range(c.K):
            if q != root:
                path = nx.shortest_path(prunedWasnNX, source=q, target=root)
                if len(path) == 2:
                    # `q` is directly connected to `u`
                    nq[q] = q  # direct neighbor
                else:
                    nq[q] = path[-2]  # (grand-)parent of `q`

        # Compute `Qqup_k` for each node
        Qqup_k = [
            int(np.sum(
                np.sum(obsMat[[q] + list(upstNodes[q]), :], axis=0) >= 2
            ))
            if len(upstNodes[q]) > 0
            else int(np.sum(obsMat[q, :]))
            for q in range(c.K)
        ]

        # Compute TI-dMWF dimension for each node
        hMk = [
            int(c.Mk[q] + np.sum([
                Qqup_k[i] for i in upstNeighbors[q]
            ])) for q in range(c.K)
        ]
        
        return TreeWASN(
            nodesPos=nodesPos,
            root=root,
            aMat=adjMat,
            upNodes=upstNodes,
            upNeighs=upstNeighbors,
            downNodes=downNodes,
            downNeighs=downNeighbors,
            levels=treeStructure,
            pathsToRoot=pathToRoot,
            nq=nq,
            Qqup_k=Qqup_k,
            hMk=hMk,
        )
    
    def random_scm(self, n, r):
        """Generate a random Hermitian, positive-semidefinite n x n matrix with rank r."""
        c = self.cfg
        if c.domain == 'wola':
            size = (c.nPosFreqs, n, r)
        else:   
            size = (n, r)
        mat = self.cfg.randmat(size)
        return mat @ herm(mat)


    def load_sound_file(self, file_path, noise=False):
        """Load a sound file and resample it to the desired sampling frequency."""
        c = self.cfg
        soundData, fsRead = sf.read(file_path, dtype='float32')
        if fsRead != c.fs:
            # Resample the signal to the desired sampling frequency
            soundData = resample(soundData, fsRead, c.fs)
        # Apply high-pass filter (get rid of potential low-frequency hum from low-quality dataset)
        # soundData = butter_highpass_filter(soundData, 0.01, 5)
        # Normalize the signal
        soundData /= np.amax(np.abs(soundData))  # Normalize
        soundData -= np.mean(soundData)  # Remove DC offset
        if c.friendlyVoiceActivity and c.scmEstimation == 'online' and\
            c.desSigType == 'speech' and not noise:
            # Compute local VAD
            vad = self.compute_vad(soundData)
            # Get rid of initial pause (0-VAD)
            start_idx = np.argmax(vad > 0)
            soundData = soundData[start_idx:]
            # Make fade in window
            durPhase = int(c.fs * c.friendlyVoiceActivityCycleDuration)
            durFade = durPhase // 10
            fade_in = np.linspace(1, 0, durFade)[::-1]  # Reverse to fade in
            # Apply fade in
            soundData[:durFade] *= fade_in
            # Truncate signal if too long
            if len(soundData) > durPhase // 2:
                # Make fade out window
                fade_out = np.linspace(1, 0, durFade)
                # Truncate
                soundData = soundData[:durPhase // 2]
                soundData[-durFade:] *= fade_out
            # Pad with zeros
            if len(soundData) < durPhase:
                soundData = np.pad(soundData, (0, durPhase - len(soundData)), mode='constant')
        return soundData


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = sig.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = sig.filtfilt(b, a, data)
    return y


def get_adjacency_matrix(cfg: Parameters, nodePos):
    # Create connection matrix (adjacency matrix) based on node positions
    dists = np.zeros((cfg.K, cfg.K))
    for k in range(cfg.K):
        for q in range(cfg.K):
            dists[k, q] = np.linalg.norm(nodePos[k, :] - nodePos[q, :])
    # Make binary
    cd = copy.deepcopy(cfg.commDist)
    aMat = np.where(dists < cd, 1, 0)
    # Check if the graph is connected
    while not np.linalg.matrix_power(aMat, cfg.K).all():
        cd += 0.1
        aMat = np.where(dists < cd, 1, 0)
    # Diagonal should be zero
    np.fill_diagonal(aMat, 0)

    return aMat, cd


def mmut_pruning(graph: nx.Graph, root):
    """
    Prune the graph to a tree with maximum number of branches at the root node.
    """
    # Step 1: Identify the fixed edges E_f
    E_f = [
        (u, v, data['weight'])
        for u, v, data in graph.edges(data=True) if u == root or v == root
    ]
    
    # Step 2: Create a subgraph with just the fixed edges
    Fg = nx.Graph()
    Fg.add_weighted_edges_from(E_f)
    
    # Step 3: Initialize the MST with the fixed edges subgraph
    mst = nx.Graph()
    mst.add_nodes_from(graph.nodes)
    mst.add_edges_from(Fg.edges(data=True))
    
    # Step 4: Sort all edges by weight
    sorted_edges = sorted(graph.edges(data=True), key=lambda x: x[2]['weight'])
    
    # Step 5: Add remaining edges ensuring no cycles (using Kruskal's algorithm logic)
    for u, v, data in sorted_edges:
        if mst.has_edge(u, v):
            continue  # Skip fixed edges already in MST
        if not nx.has_path(mst, u, v):
            mst.add_edge(u, v, **data)
    
    return mst


def get_tree_structure(G: nx.Graph, root):
    # Compute shortest path distances from root
    dist = nx.single_source_shortest_path_length(G, root)
    
    # Get the tree structure (list of leaves to root, per level)
    tree_structure = [[] for _ in range(max(dist.values()) + 1)]
    for q in G.nodes:
        tree_structure[dist[q]].append(q)
    tree_structure.reverse()  # Reverse the order to have root at the end
    
    return tree_structure


def get_downstream_nodes(G: nx.Graph, root):
    nNodes = G.number_of_nodes()
    downstreamNeighbors = [None] * nNodes  # Initialize all entries to None
    downstreamNodes = [[] for _ in range(nNodes)]

    visited = set()
    queue = deque([root])
    visited.add(root)

    while queue:
        current = queue.popleft()
        for neighbor in G.neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                downstreamNeighbors[neighbor] = current  # current is "toward root" for neighbor
                queue.append(neighbor)
    
    # Find downstream nodes for each node
    for k in G.nodes:
        currIdx = k
        while downstreamNeighbors[currIdx] is not None:
            downstreamNodes[k].append(downstreamNeighbors[currIdx])
            currIdx = downstreamNeighbors[currIdx]

    return downstreamNodes, downstreamNeighbors


def get_upstream_nodes(G: nx.Graph, root):
    """GPT"""
    # Compute shortest path distances from root
    dist = nx.single_source_shortest_path_length(G, root)
    
    upstreamNodes = [None for _ in range(len(G.nodes))]
    upstreamNeighbors = [None for _ in range(len(G.nodes))]
    for q in G.nodes:
        # Nodes "away" from root from the perspective of q:
        nodes_away = []

        # BFS from q, but only go to nodes with higher distance from root
        visited = set()
        stack = [q]
        
        while stack:
            node = stack.pop()
            visited.add(node)
            
            for neighbor in G.neighbors(node):
                if neighbor in visited:
                    continue
                if dist.get(neighbor, float('inf')) > dist[q]:
                    stack.append(neighbor)
                    nodes_away.append(neighbor)

        # Filter out q and root explicitly, if needed
        upstreamNodes[q] = np.sort([n for n in nodes_away if n != q and n != root])
        # Get the upstream neighbors of q
        upstreamNeighbors[q] = np.sort([n for n in G.neighbors(q) if n in upstreamNodes[q]])
    
    return upstreamNodes, upstreamNeighbors


# def single_update_scm(RssPrev, RnnPrev, ssH, nnH, beta, vad=None):
#     """Update the SCMs using the online estimation formula."""
#     Rss = copy.deepcopy(RssPrev)  # by default, copy the previous SCM
#     Rnn = copy.deepcopy(RnnPrev)  # by default, copy the previous SCM
#     if vad is not None:
#         if any(vad):  # <-- VOICE ACTIVITY ON
#             Rss = beta * RssPrev + (1 - beta) * ssH
#             # no update for noise SCM if VAD is False
#         else:  # <-- VOICE ACTIVITY OFF
#             Rss = beta * RssPrev + (1 - beta) * ssH
#             # no update for speech+noise/speech-only SCM if VAD is False
#             if RnnPrev is not None:
#                 # Rnn = beta * RnnPrev + (1 - beta) * yyH
#                 Rnn = beta * RnnPrev + (1 - beta) * nnH
#     else:
#         Rss = beta * RssPrev + (1 - beta) * ssH
#         if RnnPrev is not None:
#             Rnn = beta * RnnPrev + (1 - beta) * nnH
#     return Rss, Rnn

# import numpy as np

def single_update_scm(Rss, Rnn, *, ssH, nnH, beta, vad=None, inplace=True, forceRyyUp=False):
    """
    Online SCM update:
      Rss <- beta * Rss + (1 - beta) * ssH
      Rnn <- beta * Rnn + (1 - beta) * nnH   (only when VAD == False or VAD is None)

    Parameters
    ----------
    Rss, Rnn : np.ndarray or None
        Current SCMs (will be updated in-place if `inplace=True`).
    ssH, nnH : np.ndarray
        New instantaneous SCM estimates.
    beta : float
        Forgetting factor in [0, 1).
    vad : None, bool, or array-like
        If provided, noise SCM is updated **only when** np.any(vad) is False.
        (Speech SCM is always updated, matching your original code.)
    inplace : bool
        If True, modify Rss/Rnn in place; otherwise return new arrays.

    Returns
    -------
    (Rss_out, Rnn_out)
    """
    alpha = 1.0 - beta

    # Decide outputs (views or fresh arrays)
    Rss_out = Rss if (inplace and Rss is not None) else (None if Rss is None else Rss.copy())
    Rnn_out = Rnn if (inplace and Rnn is not None) else (None if Rnn is None else (None if Rnn is None else Rnn.copy()))

    # --- Rss update ---
    update_rss = ((Rss_out is not None) and (np.any(vad))) or forceRyyUp
    # update_rss = (Rss_out is not None)
    if update_rss:
        # In-place axpby: Rss = beta*Rss + alpha*ssH
        Rss_out *= beta
        Rss_out += alpha * ssH  # (one temp for ssH*alpha; cheap vs deepcopy)

    # --- Rnn update: only when VAD is False (or VAD not provided) ---
    update_rnn = (Rnn_out is not None) and (vad is None or not np.any(vad))
    if update_rnn:
        Rnn_out *= beta
        Rnn_out += alpha * nnH

    return Rss_out, Rnn_out


def single_update_scm_inplace(Rss, Rnn, ssH, nnH, beta, vad=None):
    """In-place exponential SCM updates."""
    # Note: use ufuncs with `out=` to avoid temporaries.
    if vad is not None:
        if bool(np.any(vad)):  # voice active
            # Rss = beta*Rss + (1-beta)*ssH
            np.multiply(Rss, beta, out=Rss)
            np.add(Rss, (1.0 - beta) * ssH, out=Rss)
            # Rnn unchanged
        else:                   # voice inactive
            if Rnn is not None:
                np.multiply(Rnn, beta, out=Rnn)
                np.add(Rnn, (1.0 - beta) * nnH, out=Rnn)
            # Rss unchanged
    else:
        np.multiply(Rss, beta, out=Rss)
        np.add(Rss, (1.0 - beta) * ssH, out=Rss)
        if Rnn is not None:
            np.multiply(Rnn, beta, out=Rnn)
            np.add(Rnn, (1.0 - beta) * nnH, out=Rnn)
    # return Rss, Rnn


def compute_connectivity(cfg: Parameters, aMat):
    """Compute connectivity as defined for DANSE+.""" 
    n1s = np.sum(aMat)
    n1s_fc = cfg.K * (cfg.K - 1)  # number of 1's in adjacency matrix for full connectivity
    n1s_mc = 2 * cfg.K  # number of 1's in adjacency matrix for minimum connectivity
    return (n1s - n1s_mc) / (n1s_fc - n1s_mc)

# ----------------------------
# Geometry helpers
# ----------------------------
def point_to_plane_distance(p, p0, n):
    """Distance from point p to plane (p0, n)."""
    return abs(np.dot(p - p0, n))

def project_point_to_plane(p, p0, n):
    """Orthogonal projection of p onto plane (p0, n)."""
    return p - np.dot(p - p0, n) * n

def point_in_polygon_3d(p, poly, n):
    """
    Check if point p lies inside a convex 3D polygon poly.
    Uses half-space tests.
    """
    m = poly.shape[1]
    for i in range(m):
        a = poly[:, i]
        b = poly[:, (i + 1) % m]
        edge = b - a
        outward = np.cross(edge, n)
        if np.dot(p - a, outward) > 1e-9:
            return False
    return True

def point_to_segment_distance(p, a, b):
    """Distance from point p to segment a-b."""
    ab = b - a
    t = np.dot(p - a, ab) / np.dot(ab, ab)
    t = np.clip(t, 0.0, 1.0)
    proj = a + t * ab
    return np.linalg.norm(p - proj)

def point_to_polygon_distance(p, poly):
    """
    Minimum Euclidean distance from point p to a convex polygon poly (3 x N).
    """
    # Plane normal
    n = np.cross(poly[:, 1] - poly[:, 0], poly[:, 2] - poly[:, 0])
    n = n / np.linalg.norm(n)

    # Plane distance
    d_plane = point_to_plane_distance(p, poly[:, 0], n)
    p_proj = project_point_to_plane(p, poly[:, 0], n)

    # If projection lies inside polygon, plane distance is the answer
    if point_in_polygon_3d(p_proj, poly, n):
        return d_plane

    # Otherwise, distance to closest edge
    d_edges = [
        point_to_segment_distance(p, poly[:, i], poly[:, (i + 1) % poly.shape[1]])
        for i in range(poly.shape[1])
    ]
    return min(d_edges)

def sample_position_from_room(
    walls: list[pra.libroom.Wall],
    min_dist,
    rng=None,
    max_tries=100_000,
):
    """
    Sample a random position inside a pyroomacoustics room
    with minimum distance min_dist to ALL walls.
    """
    rng = np.random.default_rng() if rng is None else rng

    # Conservative bounding box from wall vertices
    all_pts = np.hstack([w.corners for w in walls])
    mins = all_pts.min(axis=1) + min_dist
    maxs = all_pts.max(axis=1) - min_dist

    if np.any(mins >= maxs):
        raise ValueError("No feasible region for this min_dist.")

    for _ in range(max_tries):
        p = rng.uniform(mins, maxs)

        ok = True
        for w in walls:
            if point_to_polygon_distance(p, w.corners) < min_dist:
                ok = False
                break

        if ok:
            return p

    raise RuntimeError(
        f"Could not sample a valid position in {max_tries} attempts."
    )

def compute_observability_matrix(
    P_ms,
    P_self,
    T_sir_db=-10.0,
    T_snr_db=0.0,
    use_network_gate=False,
    delta_db=30.0,
    eps=1e-12,
):
    """
    Compute an M x S observability matrix.

    Parameters
    ----------
    P_ms : ndarray, shape (M, S)
        Oracle power of each source s at microphone m (speech + noise sources).
        Must be linear power (not dB).

    P_self : ndarray, shape (M,)
        Microphone self-noise power at each microphone (linear scale).

    T_sir_db : float, optional
        SIR threshold in dB.
        Default: -10 dB.

    T_snr_db : float, optional
        Self-noise SNR threshold in dB.
        Default: 0 dB.

    use_network_gate : bool, optional
        Whether to apply the network-relative exposure gate.
        Default: False.

    delta_db : float, optional
        Network-relative threshold Δ in dB.
        A source must be within Δ dB of its strongest microphone.
        Only used if use_network_gate=True.
        Default: 30 dB.

    eps : float, optional
        Small constant to avoid division by zero.
        Default: 1e-12.

    Returns
    -------
    observable : ndarray, shape (M, S), dtype=bool
        Boolean observability matrix.
        observable[m, s] == True iff source s is observable at mic m.
    """

    P_ms = np.asarray(P_ms, dtype=float)
    P_self = np.asarray(P_self, dtype=float)

    M, S = P_ms.shape

    # Total source power at each mic
    P_sources_total = np.sum(P_ms, axis=1)  # shape (M,)

    # Interference + self-noise power for each (m, s)
    # P_{m,¬s} = sum_{s'≠s} P_{m,s'} + P_self[m]
    P_interf = (
        P_sources_total[:, None]
        - P_ms
        + P_self[:, None]
    )

    # --- SIR gate ---
    SIR_db = 10.0 * np.log10((P_ms + eps) / (P_interf + eps))
    sir_ok = SIR_db >= T_sir_db

    # --- Self-noise SNR gate ---
    SNR_db = 10.0 * np.log10((P_ms + eps) / (P_self[:, None] + eps))
    snr_ok = SNR_db >= T_snr_db

    # Combine mandatory gates
    observable = sir_ok & snr_ok

    # --- Optional network-relative gate ---
    if use_network_gate:
        # Max power per source across microphones
        P_max_per_source = np.max(P_ms, axis=0)  # shape (S,)

        rel_db = 10.0 * np.log10((P_ms + eps) / (P_max_per_source[None, :] + eps))
        net_ok = rel_db >= -delta_db

        observable &= net_ok

    return observable
