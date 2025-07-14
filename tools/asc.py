# Contents of package:
# Classes and functions related to the acoustic scenario for dMWF.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import copy
import time
import numpy as np
import networkx as nx
import soundfile as sf
from pathlib import Path
import scipy.signal as sig
from .base import Parameters
from resampy import resample
import pyroomacoustics as pra
from collections import deque
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

@dataclass
class StaticScenarioParameters:
    """A dataclass for static scenario parameters."""
    nodesPos: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    sensorsPos: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    speechSourcesPos: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    noiseSourcesPos: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    obsMat: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    Qkq: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    rirs: list[np.ndarray] = field(default_factory=list)  # room impulse responses
    aMat: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))  # adjacency matrix
    trees: list = field(default_factory=list)  # list of TreeWASN objects


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
    td: dict[np.ndarray] = field(default_factory=dict)
    wd: dict[np.ndarray] = field(default_factory=dict)


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
    obsMat: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))  # observability matrix
    Qkq: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))  # number of sources in common between nodes
    oQkq: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))  # number of sources observed by each node

    def setup(self):
        c = self.cfg
        if c.domain == 'wola':
            out = self.setup_wola_domain()
        if 'time' in c.domain:
            out = self.setup_time_domain()
        
        if c.observability == 'foss':
            self.oQkq = np.full((c.K, c.K), c.Q)
            self.Qkq = np.full((c.K, c.K), c.Q)
        elif c.observability == 'poss':            
            # Compute the number of sources in common between nodes k and q
            self.Qkq = np.zeros((c.K, c.K), dtype=int)
            for k in range(c.K):
                for q in range(c.K):
                    self.Qkq[k, q] = np.sum(
                        self.obsMat[k, :] * self.obsMat[q, :]
                    )
            # Number of sources useful for fusion matrix computation for node q
            self.oQkq = np.zeros((c.K, c.K), dtype=int)
            for ii in range(c.Q):
                for q in range(c.K):
                    if self.obsMat[q, ii]:  # node q observes source ii
                        for k in range(c.K):
                            if np.any([
                                self.obsMat[qp, ii]
                                for qp in range(c.K)
                                if qp != q and self.Qkq[k, qp] > 0  # qp criterion includes k
                            ]):
                                # If any other node that observe source ii
                                # observe a source in common with node k, count
                                # this source as observed by node k
                                self.oQkq[q, k] += 1
            assert np.all(self.oQkq >= self.Qkq), \
                "oQkq must be greater than or equal to Qkq."
            pass
        
        return out

    def setup_time_domain(self):
        """Setup the acoustic scenario in the time domain."""
        c = self.cfg
        # Initialize matrices randomly
        Amat = c.randmat((c.M, c.Qd))
        Bmat = c.randmat((c.M, c.Qn))
        cAmat = [copy.deepcopy(Amat) for _ in range(c.K)]
        cBmat = [copy.deepcopy(Bmat) for _ in range(c.K)]
        slat = c.randmat((c.Qd, c.N))
        nlat = c.randmat((c.Qn, c.N))
        pows = np.mean(np.abs(slat) ** 2, axis=1)
        pown = np.mean(np.abs(nlat) ** 2, axis=1)
        # Compute steering matrices
        if c.observability == 'foss':
            self.oQkq = np.full(c.K, c.Q)
            self.Qkq = np.full((c.K, c.K), c.Q)
        elif c.observability == 'poss':
            # Do not differentiate between global and local sources, 
            # randomly generate observability pattern
            self.obsMat = np.zeros((c.K, c.Q))
            # def inadequate(om):
            #     return np.any(np.sum(om, axis=1) == 0) | np.any(np.sum(om, axis=0) == 0)
            
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
                        Amat[c.Mk * k:c.Mk * (k + 1), s] = 0
                for n in range(c.Qn):
                    if self.obsMat[k, c.Qd + n] == 0:
                        Bmat[c.Mk * k:c.Mk * (k + 1), n] = 0
            for k in range(c.K):
                # "Global" Amat and Bmat matrices
                # List of sources that are either not observed by node k, or
                # not observed by any other node
                idxUncorr_A, idxUncorr_B = [], []
                for s in range(c.Qd):
                    if self.obsMat[k, s] == 0 or np.sum(self.obsMat[:, s]) == 1:
                        idxUncorr_A.append(s)
                for n in range(c.Qn):
                    if self.obsMat[k, c.Qd + n] == 0 or np.sum(self.obsMat[:, c.Qd + n]) == 1:
                        idxUncorr_B.append(n)
                if len(idxUncorr_A) > 0:
                    cAmat[k][:, idxUncorr_A] = 0
                if len(idxUncorr_B) > 0:
                    cBmat[k][:, idxUncorr_B] = 0

        # Compute signals
        s = Amat @ slat
        n = Bmat @ nlat
        v = c.randmat((c.M, c.N)) * np.mean(pows) * c.selfNoiseFactor  # small self-noise
        n += v  # add self-noise to noise signal

        # Compute the SCMs
        if c.scmEstimation == 'oracle':
            # For oracle SCM estimation, we assume perfect knowledge of the
            # source and noise steering matrices
            Gam_s = np.diag(pows)
            Gam_n = np.diag(pown)
            Rss = Amat @ Gam_s @ Amat.conj().T
            Rnn = Bmat @ Gam_n @ Bmat.conj().T
            Rvv = np.eye(c.M) * np.mean(pows) * c.selfNoiseFactor  # small self-noise
            Rnn += Rvv  # add self-noise to noise SCM
        elif c.scmEstimation == 'batch':
            # Batch SCM estimation based on actual signals
            Rss = s @ s.conj().T / c.N
            Rnn = n @ n.conj().T / c.N
        
        # Complete signal SCM
        Ryy = Rss + Rnn

        return Ryy, Rss, Rnn, s, n
    
    def setup_wola_domain(self):
        """Setup the acoustic scenario in the WOLA domain."""
        c = self.cfg
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

        # Get latent signals
        self.latentSpeech = self.gen_latent_speech(n=c.Qd)
        self.latentNoise = self.get_latent_noise(n=c.Qn)
        
        # Scale the latent noise signals to match the target SNR
        if c.latentSNR is not None:
            # Scale the latent noise signals to match the target SNR
            nPower = np.mean(np.abs(self.latentNoise) ** 2)
            sPower = np.mean(np.abs(self.latentSpeech) ** 2)
            if nPower > 0 and sPower > 0:
                currentSNR = 10 * np.log10(sPower / nPower)
                scalingFactor = 10 ** ((c.latentSNR - currentSNR) / 20)
                self.latentNoise /= scalingFactor

        # Prepare containers
        self.nodes = [
            Node(
                idx=k,
                td={
                    'y': np.zeros((c.Mk, c.N)),
                    's': np.zeros((c.Mk, c.N)),
                    'n': np.zeros((c.Mk, c.N)),
                    'sn': np.zeros((c.Mk, c.N)),
                    'sIndiv': np.zeros((c.Qd, c.Mk, c.N)),
                    'nIndiv': np.zeros((c.Qn, c.Mk, c.N)),
                }
            )
            for k in range(c.K)
        ]

        # Generate the WASN parameters
        p: StaticScenarioParameters = self.define_static_scenario(room)
        self.apply_static_scenario(p)
        # Store the parameters for later use
        self.store_scenario_parameters(p)
        
        # Self-noise addition
        for k in range(c.K):
            # Generate self-noise at correct SNR
            sn = c.randmat((c.Mk, c.N), makeComplex=False)
            snPower = np.mean(np.abs(sn) ** 2)
            sPower = np.mean(np.abs(self.nodes[k].td['s']) ** 2)
            sn *= np.sqrt(c.selfNoiseFactor * sPower / snPower)
            self.nodes[k].td['sn'] = sn
            self.nodes[k].td['n'] += self.nodes[k].td['sn']
            self.nodes[k].td['y'] = self.nodes[k].td['n'] + self.nodes[k].td['s']

        print("Acoustic environment generated successfully, computing SCMs...")
        # Compute SCMs (from steering matrices if oracle, from signals if batch)
        return self.compute_scms()

    def compute_scms(self):
        """Compute the SCMs for the acoustic scenario."""
        c = self.cfg
        # Compute centralized signals STFT
        stack = dict()
        for st in ['s', 'n', 'sn']:
            tmp = np.vstack(
                [self.nodes[k].td[st] for k in range(c.K)]
            )
            stack[st] = c.get_stft(tmp)
        stack['n'] = stack['n'] + stack['sn']  # Add self-noise to noise signal
        print(f'{c.T} s of signals = {stack["s"].shape[-1]} STFT frames.')
        # Stack the signals
        if c.scmEstimation == 'oracle':
            # Compute FFT of RIRs
            steeringMats = np.zeros((c.M, c.Q, c.nPosFreqs), dtype=complex)
            for ii in range(c.Q):
                rirs = np.array([self.rirs[m][ii] for m in range(c.M)])
                tmp = np.fft.rfft(rirs, n=c.nfft, axis=-1)
                if c.singleLine is not None:
                    # If singleLine is set, only use the specified frequency line
                    tmp = tmp[:, [c.singleLine]]
                # Set the steering vectors of nodes that do not observe source ii to zero
                for q in np.where(self.obsMat[:, ii] == 0)[0]:
                    tmp[c.Mk * q:c.Mk * (q + 1), :] = 0
                steeringMats[:, ii, :] = tmp
            # Compute STFTs of latent signals
            slatSTFT = c.get_stft(self.latentSpeech)
            nlatSTFT = c.get_stft(self.latentNoise)
            power_s = np.mean(np.abs(slatSTFT) ** 2, axis=-1)
            power_n = np.mean(np.abs(nlatSTFT) ** 2, axis=-1)
            power_v = np.mean(np.abs(stack['sn']) ** 2, axis=-1)
            # Compute the SCMs
            Rss = np.zeros((c.nPosFreqs, c.M, c.M), dtype=complex)
            Rnn = np.zeros((c.nPosFreqs, c.M, c.M), dtype=complex)
            Ryy = np.zeros((c.nPosFreqs, c.M, c.M), dtype=complex)
            for kappa in range(c.nPosFreqs):
                Rsslat = np.diag(power_s[:, kappa])
                Rnnlat = np.diag(power_n[:, kappa])
                Rss[kappa, ...] = steeringMats[:, :c.Qd, kappa] @ Rsslat @ steeringMats[:, :c.Qd, kappa].conj().T
                Rnn[kappa, ...] = steeringMats[:, c.Qd:, kappa] @ Rnnlat @ steeringMats[:, c.Qd:, kappa].conj().T
                Rvv = np.diag(power_v[:, kappa])
                Ryy[kappa, ...] = Rss[kappa, ...] + Rnn[kappa, ...] + Rvv

        elif c.scmEstimation == 'batch':
            # Compute the SCMs
            nFrames = stack['s'].shape[-1]
            Rss = np.einsum('ijk,ljk->jil', stack['s'], stack['s'].conj()) / nFrames
            Rnn = np.einsum('ijk,ljk->jil', stack['n'], stack['n'].conj()) / nFrames
            # Complete signal SCM
            Ryy = Rss + Rnn

        return Ryy, Rss, Rnn, stack['s'], stack['n']

    def plot(self):
        """Export the environment to a TXT file and to a plot."""
        c = self.cfg
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
                axes.plot(self.nodesPos[k, 0], self.nodesPos[k, 1], 'ro', markersize=2)
                axes.text(self.nodesPos[k, 0] + c.nodeRadius, self.nodesPos[k, 1] + c.nodeRadius, str(k+1))
            # Plot the sensors
            for k in range(c.K):
                for m in range(c.Mk):
                    axes.plot(
                        self.sensorsPos[k * c.Mk + m, 0],
                        self.sensorsPos[k * c.Mk + m, 1],
                        'ko', markersize=2
                    )
            # Add circle around the nodes 
            for k in range(c.K):
                circle = plt.Circle(
                    (self.nodesPos[k, 0], self.nodesPos[k, 1]),
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
                    self.speechSourcesPos[ii, 0],
                    self.speechSourcesPos[ii, 1], 'go')
                axes.text(
                    self.speechSourcesPos[ii, 0],
                    self.speechSourcesPos[ii, 1], f'S{ii+1}')
            for ii in range(c.Qn):
                axes.plot(
                    self.noiseSourcesPos[ii, 0],
                    self.noiseSourcesPos[ii, 1], 'rd')
                axes.text(
                    self.noiseSourcesPos[ii, 0],
                    self.noiseSourcesPos[ii, 1], f'N{ii+1}')
            # Plot the observability matrix
            axes.grid()
            axes.set_aspect('equal')
            axes.set_title(f"Acoustic environment")
            figObs.tight_layout()

            # Create a copy of the figure without observability lines
            fig = copy.deepcopy(figObs)

            # Add lines between nodes and sources to indicate observability
            for k in range(c.K):
                for ii in range(c.Qd + c.Qn):
                    if self.obsMat[k, ii] == 1:
                        if ii < c.Qd:
                            axes.plot(
                                [self.nodesPos[k, 0], self.speechSourcesPos[ii, 0]],
                                [self.nodesPos[k, 1], self.speechSourcesPos[ii, 1]],
                                'g:',
                                alpha=0.5
                            )
                        else:
                            axes.plot(
                                [self.nodesPos[k, 0], self.noiseSourcesPos[ii - c.Qd, 0]],
                                [self.nodesPos[k, 1], self.noiseSourcesPos[ii - c.Qd, 1]],
                                'r:',
                                alpha=0.5
                            )
            axes.set_title(f"Observabilities")
        else:
            print("Plotting not implemented for 3D environments.")
        return fig, figObs

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

    def gen_latent_speech(self, n: int) -> list[str]:
        """Generate speech signals using files from the database."""
        c = self.cfg
        if c.desSigType == 'random':
            # Generate white noise signals
            return c.randmat((n, c.N), makeComplex=False)
        elif c.desSigType == 'speech':
            # Recursively search for all .wav files in the database path
            path = Path(c.speechDatabasePath)
            files = list(path.rglob("*.wav")) + list(path.rglob("*.flac"))
            if len(files) == 0:
                raise ValueError("No files found in the database path.")
            nSamples = c.N
            x = np.zeros((n, nSamples), dtype='float32')
            alreadyUsed = []
            for ii in range(n):
                # Randomly select one file from the list
                fr = files[np.random.randint(0, len(files), 1)[0]].name
                while fr in alreadyUsed:
                    fr = files[np.random.randint(0, len(files), 1)[0]].name
                alreadyUsed.append(fr)
                if '-' in fr:
                    prefix = fr.split("-")[0]
                elif '_' in fr:
                    prefix = fr.split("_")[0]
                # Load file 
                speech = load_sound_file(files[ii], desFs=c.fs)
                idx = 0
                while len(speech) < nSamples:
                    # Find another file with the same prefix
                    filesSamePrefix = [f for f in files if f.name.startswith(prefix) and f.name not in alreadyUsed]
                    if len(filesSamePrefix) == 0:
                        print(f"No more files found with the same prefix as {fr}. Changing prefix at {len(speech)} samples (= {len(speech)/c.fs:.2f} s)...")
                        # Randomly select one file from the list
                        fr = files[np.random.randint(0, len(files), 1)[0]].name
                        while fr in alreadyUsed:
                            fr = files[np.random.randint(0, len(files), 1)[0]].name
                        if '-' in fr:
                            prefix = fr.split("-")[0]
                        elif '_' in fr:
                            prefix = fr.split("_")[0]
                        filesSamePrefix = [f for f in files if f.name.startswith(prefix) and f.name not in alreadyUsed]
                        idx = 0
                    else:
                        # Randomly select one file from the list
                        fr = filesSamePrefix[np.random.randint(0, len(filesSamePrefix), 1)[0]].name
                        while fr in alreadyUsed:
                            fr = filesSamePrefix[np.random.randint(0, len(filesSamePrefix), 1)[0]].name
                    alreadyUsed.append(fr)
                    # Concat the file to the speech signal
                    newSig = load_sound_file(filesSamePrefix[idx], desFs=c.fs)
                    speech = np.concatenate((speech, newSig))
                    idx += 1
                x[ii, :] = speech[:nSamples]
            return x
        
    def get_latent_noise(self, n: int) -> np.ndarray:
        """Generate noise signals."""
        c = self.cfg
        if c.noiseSigType == 'random':
            # Generate white noise signals
            return c.randmat((n, c.N), makeComplex=False)
        elif c.noiseSigType == 'babble':
            out = np.zeros((n, c.N), dtype='float32')
            # Select babble noise files from the babble database
            allBabbles = list(Path(c.babbleDatabasePath).rglob("babble*.wav"))
            for ii in range(n):
                # Randomly select one file from the babble database
                if ii >= len(allBabbles):
                    raise ValueError(f"Not enough babble files in the database (asked for {n}, found {len(allBabbles)}).")
                else:
                    babbleFile = allBabbles[ii]
                tmp = load_sound_file(babbleFile, desFs=c.fs)
                while len(tmp) < c.N:
                    # If the file is too short, concatenate it with another file
                    tmp2 = load_sound_file(babbleFile, desFs=c.fs)
                    tmp = np.concatenate((tmp, tmp2))
                out[ii, :] = tmp[:c.N]
            return out

    def apply_static_scenario(self, p: StaticScenarioParameters, smIdx: list[int] = None):
        c = self.cfg  # Configuration object
        if smIdx is None:
            smIdx = np.array([0, c.N - 1], dtype=int)  # Default indices for the whole signal
        # Apply the room impulse responses to the latent signals
        for k in range(c.K):
            # Get the indices of the microphones for this node
            micIdx = np.arange(k * c.Mk, (k + 1) * c.Mk)
            # Apply the room impulse responses to the latent signals
            for ii in range(c.Qd):
                # ---------------------------------------
                if p.obsMat[k, ii] == 0:
                    continue  # Skip if the node does not observe the source
                # ---------------------------------------
                for jj, m in enumerate(micIdx):
                    tmp = sig.fftconvolve(
                        self.latentSpeech[ii, smIdx[0]:smIdx[1]],
                        p.rirs[m][ii],
                    )
                    self.nodes[k].td['sIndiv'][ii, jj, smIdx[0]:smIdx[1]] = tmp[
                        :int(smIdx[1] - smIdx[0])
                    ]
            for ii in range(c.Qn):
                # ---------------------------------------
                if p.obsMat[k, c.Qd + ii] == 0:
                    continue  # Skip if the node does not observe the source
                # ---------------------------------------
                for jj, m in enumerate(micIdx):
                    tmp = sig.fftconvolve(
                        self.latentNoise[ii, smIdx[0]:smIdx[1]],
                        p.rirs[m][c.Qd + ii]
                    )
                    self.nodes[k].td['nIndiv'][ii, jj, smIdx[0]:smIdx[1]] = tmp[
                        :int(smIdx[1] - smIdx[0])
                    ]
            self.nodes[k].td['s'] = np.sum(self.nodes[k].td['sIndiv'], axis=0)
            self.nodes[k].td['n'] = np.sum(self.nodes[k].td['nIndiv'], axis=0)

    def define_static_scenario(self, room: pra.ShoeBox) -> StaticScenarioParameters:
        c = self.cfg  # Configuration object
        rd = [c.roomLength, c.roomWidth, c.roomHeight]
        # Generate node positions
        nodesPos = np.random.rand(c.K, 3) *\
            (np.array(rd) - 2 * c.minDistFromWall) + c.minDistFromWall
        
        # Generate sensor positions around node position
        sensorsPos = np.zeros((c.K, c.Mk, 3))
        for k in range(c.K):
            sensorsPos[k, :, :] = nodesPos[k, :] +\
                np.random.randn(c.Mk, 3) * (c.nodeRadius / 2)
        # Flatten the sensor positions
        sensorsPos = sensorsPos.reshape(c.K * c.Mk, 3)
        
        if c.onPlane:
            # Ensure that the nodes are on the same plane
            zPlane = 1.3 if c.roomHeight > 1.3 else c.roomHeight / 2.0
            nodesPos[:, 2] = zPlane
            sensorsPos[:, 2] = zPlane

        # Generate source positions
        speechSourcesPos = np.zeros((c.Qd, 3))
        for ii in range(c.Qd):
            attempt = np.random.rand(3) *\
                (np.array(rd) - 2 * c.minDistFromWall) + c.minDistFromWall
            if c.onPlane:
                attempt[2] = zPlane
            counter = 0
            while np.linalg.norm(attempt - nodesPos, axis=1).min() < c.minDistNodeSource:
                print(f"Desired source {ii + 1}/{c.Qd} too close to a node, generating a new position ({counter+1}-th trial)...", end='\r')
                # If the source is too close to a node, generate a new position
                attempt = np.random.rand(3) *\
                    (np.array(rd) - 2 * c.minDistFromWall) + c.minDistFromWall
                if c.onPlane:
                    attempt[2] = zPlane
                counter += 1
            # Store the position of the source
            speechSourcesPos[ii, :] = attempt

        # Generate noise source positions
        noiseSourcesPos = np.zeros((c.Qn, 3))
        for ii in range(c.Qn):
            attempt = np.random.rand(3) *\
                (np.array(rd) - 2 * c.minDistFromWall) + c.minDistFromWall
            if c.onPlane:
                attempt[2] = zPlane
            counter = 0
            while np.linalg.norm(attempt - nodesPos, axis=1).min() < c.minDistNodeSource or\
                np.linalg.norm(attempt - speechSourcesPos, axis=1).min() < c.minDistNodeSource:
                print(f"Noise source {ii + 1}/{c.Qd} too close to a node or a desired source, generating a new position (trial #{counter+1})...", end='\r')
                # If the source is too close to a node, generate a new position
                attempt = np.random.rand(3) *\
                    (np.array(rd) - 2 * c.minDistFromWall) + c.minDistFromWall
                if c.onPlane:
                    attempt[2] = zPlane
                counter += 1
            # Store the position of the source
            noiseSourcesPos[ii, :] = attempt
        
        # Create observability matrix based on the node and source positions
        obsMat = self.get_observability_matrix(
            nodesPos, speechSourcesPos, noiseSourcesPos
        )  # node x source
        # Compute number of common sources for each pair of nodes
        Qkq = np.array([[
            np.sum(obsMat[k, :] + obsMat[q, :] == 2)
            for q in range(c.K)
        ] for k in range(c.K)])
        if np.any(Qkq - np.diag(np.diag(Qkq)) > c.Mk) and any('dMWF' in alg for alg in c.algos):
            raise ValueError("Number of common sources exceeds number of sensors per node.")

        # For topology-independent (TI) algorithms, generate the adjacency matrix
        flagTI = any('TI' in alg for alg in c.algos)
        if flagTI:
            # Non-fully connected network
            aMat, commDist = get_adjacency_matrix(c, nodesPos)
            # Ensure target degree of connectivity
            # self.set_target_connectivity()
            print(f"Generated WASN with connectivity {c.connectivity} with comm. distance: {commDist:.2f} m")
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

        return StaticScenarioParameters(
            nodesPos=nodesPos,
            sensorsPos=sensorsPos,
            speechSourcesPos=speechSourcesPos,
            noiseSourcesPos=noiseSourcesPos,
            obsMat=obsMat,
            Qkq=Qkq,
            rirs=room.rir,
            aMat=aMat if flagTI else None,
            trees=trees if flagTI else None,
        )

    def get_observability_matrix(self, nodesPos, speechPos, noisePos) -> np.ndarray:
        """Compute the observability matrix."""
        c = self.cfg
        if c.observability == 'foss':
            # Full observability -- all nodes observe all sources
            obsMat = np.ones((c.K, c.Qd + c.Qn))
        elif c.observability == 'poss':
            # Partial observability -- nodes only observe certain sources
            if 1:
                # Do not differentiate between global and local sources, 
                # randomly generate observability pattern
                obsMat = np.zeros((c.K, c.Q))
                def inadequate(om):
                    observedDesired = np.sum(om[:, :c.Qd], axis=1) > 0
                    observedNoise = np.sum(om[:, c.Qd:], axis=1) > 0
                    oneObserver = np.sum(om, axis=0) > 0
                    return not np.all(observedDesired) or\
                        not np.all(oneObserver)
                        # not np.all(observedNoise) or not np.all(oneObserver)
                # Criterion for adequacy: at least one desired source and one noise
                # source must be observed by each node, and each source must be
                # observed by at least one node.
                # while inadequate(self.obsMat[:, :c.Qd]) or inadequate(self.obsMat[:, c.Qd:]):
                while inadequate(obsMat):
                    obsMat = np.random.randint(0, 2, (c.K, c.Q))
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
            [(k, sensorsPos[k * c.Mk:(k + 1) * c.Mk, ...]) for k in range(c.K)]
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
            int(c.Mk + np.sum([
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


def load_sound_file(file_path, desFs):
    """Load a sound file and resample it to the desired sampling frequency."""
    soundData, fsRead = sf.read(file_path, dtype='float32')
    if fsRead != desFs:
        # Resample the signal to the desired sampling frequency
        soundData = resample(soundData, fsRead, desFs)
    # Normalize the signal
    soundData /= np.amax(np.abs(soundData))  # Normalize
    soundData -= np.mean(soundData)  # Remove DC offset
    return soundData


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

