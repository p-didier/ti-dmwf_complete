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
        elif c.observability == 'poss':
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

        # ADJUST to number of microphones
        for k in range(c.K):
            self.Qkq[k, :] = [np.amin((c.Mk[k], q)) for q in self.Qkq[k, :]]
            self.oQq[k] = np.amin((c.Mk[k], self.oQq[k]))
            self.Qdk[k] = np.amin((c.Mk[k], self.Qdk[k]))

        # Cache pre-computed selection matrices for dMWF
        self.Eqps = [[None for _ in range(c.K)] for _ in range(c.K)]
        for q in range(c.K):
            for p in range(c.K):
                if p == q:
                    continue
                oQq_eff = np.amin((c.Mk[p], self.oQq[q]))
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
        
        for s in self.scenarios:
            s.get_Qdims(c)

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
            Amat = c.randmat((c.M, c.Qd))
            Bmat = c.randmat((c.M, c.Qn))
        elif c.TDsteeringMats == 'anechoic':
            # Define an anechoic scenario
            _, sensorsPos, desSourcePos, noiSourcesPos = self.define_layout()
            Amat = self.steermat_green_anechoic(sensorsPos, desSourcePos)
            Bmat = self.steermat_green_anechoic(sensorsPos, noiSourcesPos)
            if c.domain == 'time':
                Amat = np.abs(Amat)
                Bmat = np.abs(Bmat)
        
        slat = self.gen_latent_desired(n=c.Qd)
        nlat = self.get_latent_noise(n=c.Qn)
        pows = np.mean(np.abs(slat) ** 2, axis=1)
        pown = np.mean(np.abs(nlat) ** 2, axis=1)
        # Compute steering matrices
        if c.observability == 'poss':
            # Do not differentiate between global and local sources, 
            # randomly generate observability pattern
            self.obsMat = np.zeros((c.K, c.Q))
            
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
                        Amat[c.Mkc[k]:c.Mkc[k + 1], s] = 0
                for n in range(c.Qn):
                    if self.obsMat[k, c.Qd + n] == 0:
                        Bmat[c.Mkc[k]:c.Mkc[k + 1], n] = 0

        # Compute signals
        s = Amat @ slat
        n = Bmat @ nlat
        v = c.randmat((c.M, c.N)) * np.mean(pows) * c.selfNoiseFactor  # small self-noise
        n += v  # add self-noise to noise signal

        if 0:
            self.latentDesired = slat
            self.latentNoise = nlat
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
            # Generate the WASN parameters
            p: StaticScenarioParameters = self.define_static_scenario(room)
            ## MANUALLY apply static scenario
            fLineIdx = c.singleLine
            Cmat = np.zeros((c.M, c.Q), dtype=complex)
            for ii in range(c.Q):
                rirs = np.array([p.rirs[m][ii] for m in range(c.M)])
                tmp = np.fft.rfft(rirs, n=c.nfft, axis=-1)   # RIRs FFT => transfer functions
                # Only use one frequency line
                tmp = tmp[:, fLineIdx]
                # Set the steering vectors of nodes that do not observe source ii to zero
                for q in np.where(p.obsMat[:, ii] == 0)[0]:
                    tmp[c.Mkc[q]:c.Mkc[q + 1]] = 0
                Cmat[..., ii] = tmp
            s = Cmat[..., :c.Qd] @ c.get_stft(slat)[..., 0, :]
            n = Cmat[..., c.Qd:] @ c.get_stft(nlat)[..., 0, :]
            v = c.get_stft(np.real(v))[..., 0, :]
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

        return Ryy, Rss, Rnn, s, n
    
    def setup_wola_domain(self):
        c = self.cfg

        # Set up latent signals
        self.latentDesired = self.gen_latent_desired(n=c.Qd)
        self.latentNoise = self.get_latent_noise(n=c.Qn)
        # Scale the latent noise signals to match the target SNR
        if c.latentSNR is not None:
            # Scale the latent noise signals to match the target SNR
            nPower = np.mean(np.abs(self.latentNoise) ** 2)
            sPower = np.mean(np.abs(self.latentDesired) ** 2)
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
            self.setup_wola_domain_static()
        
        elif c.dynamics == 'moving':
            # Dynamic scenario with always-active sources and random source movements
            nScenarios = np.floor(c.T / c.movingEvery).astype(int)
            print(f"Setting up {nScenarios} dynamic scenarios with different sources/nodes positions.")
            
            if c.fixedObservabilities:
                omFixed = self.get_observability_matrix()  # Get the fixed observability matrix
            else:
                omFixed = None

            for ii in range(nScenarios):
                print(f"Setting up scenario {ii + 1}/{nScenarios}...")
                idxStart = ii * int(c.movingEvery * c.fs)
                idxEnd = idxStart + int(c.movingEvery * c.fs) - 1
                # Setup the static scenario
                self.setup_wola_domain_static(idxStart, idxEnd, omFixed=omFixed)

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
        
        print("Acoustic environment generated successfully, computing SCMs...")
        # Compute SCMs (from steering matrices if oracle, from signals if batch)
        return self.compute_scms()

    def setup_wola_domain_static(self, idxStart=0, idxEnd=-1, omFixed=None):
        """Setup a static acoustic scenario in the WOLA domain."""
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

        # Generate the WASN parameters
        p: StaticScenarioParameters = self.define_static_scenario(room, omFixed=omFixed)
        self.apply_static_scenario(p, smIdx=[idxStart, idxEnd])
        # Store the parameters for later use
        self.scenarios.append(p)

    def compute_scms(self):
        """Compute the SCMs for the acoustic scenario."""
        c = self.cfg

        # Compute centralized signals STFT
        stack = dict()
        for st in ['y', 's', 'n', 'sn']:
            if c.wolaMixtures_viaTD:
                tmp = np.vstack([self.nodes[k].td[st] for k in range(c.K)])
                stack[st] = c.get_stft(tmp)
            else:
                stack[st] = np.vstack([self.nodes[k].wd[st] for k in range(c.K)])
        print(f'{c.T} s of signals = {stack["y"].shape[-1]} STFT frames.')

        if c.scmEstimation == 'oracle':
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

        elif c.scmEstimation == 'batch':
            nFrames = stack['y'].shape[-1]
            Rnn = np.einsum('ijk,ljk->jil', stack['n'], stack['n'].conj())
            if c.noCrossCorrelation:
                Rss = np.einsum('ijk,ljk->jil', stack['s'], stack['s'].conj())
                Ryy = Rss + Rnn
            else:
                Ryy = np.einsum('ijk,ljk->jil', stack['y'], stack['y'].conj())
                Rss = None

        elif c.scmEstimation == 'online':
            Ryy, Rss, Rnn = None, None, None # nothing to do -- done in `algos.py`
        else:
            raise ValueError(f"Unknown SCM estimation method: {c.scmEstimation}")

        return Ryy, Rss, Rnn, stack['s'], stack['n']

    def estimate_vad(self):
        """Estimate the VAD for the latent desired signal."""
        c = self.cfg
        smoothingLength = int(np.amin((c.vadSmoothingPeriod, c.noiseONOFFperiod)) * c.fs)
        vad = np.zeros((c.Qd, c.N), dtype=bool)
        for ii in range(c.Qd):
            # Compute the energy of the signal
            energy = np.mean(np.abs(self.latentDesired[ii, :]) ** 2)
            # Compute the threshold for VAD
            threshold = c.vadThreshold * energy
            # Smooth the VAD: avoid abrupt changes between 1 and 0
            vad[ii, :] = np.convolve(
                np.abs(self.latentDesired[ii, :]) ** 2 > threshold,
                np.ones(smoothingLength) / smoothingLength,
                mode='same'
            ) > 0.5

        # Convert to VAD in STFT domain: VAD in one frame is True if at least
        # half of the samples are True.
        vadSTFT = np.zeros((c.Qd, c.nFrames), dtype=bool)
        for l in range(c.nFrames):
            idxBeg = int(l * (c.nfft - c.nhop))
            idxEnd = idxBeg + c.nfft
            for ii in range(c.Qd):
                vadSTFT[ii, l] = np.sum(vad[ii, idxBeg:idxEnd]) > c.nfft / 2

        return vad.astype(bool), vadSTFT.astype(bool)

    def plot(self):
        """Export the environment to a TXT file and to a plot."""
        c = self.cfg
        scn = self.scenarios[0]  # Use the first scenario for plotting
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
            axes.set_title(f"Acoustic environment")
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
            if 1:
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
            ax.set_xlabel('Length [m]')
            ax.set_ylabel('Width [m]')
            ax.set_zlabel('Height [m]')
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
            alreadyUsed = []
            for ii in range(n):
                # Randomly select one file from the list
                fr = files[np.random.randint(0, len(files), 1)[0]].name
                while fr[:d] in alreadyUsed:
                    fr = files[np.random.randint(0, len(files), 1)[0]].name
                alreadyUsed.append(fr[:d])
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
                        while fr[:d] in alreadyUsed:
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
                        while fr[:d] in alreadyUsed:
                            fr = filesSamePrefix[np.random.randint(0, len(filesSamePrefix), 1)[0]].name
                    alreadyUsed.append(fr[:d])
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
                tmp = load_sound_file(currFile, desFs=c.fs)
                while len(tmp) < c.N:
                    # If the file is too short, concatenate it with another file
                    tmp2 = load_sound_file(currFile, desFs=c.fs)
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
        # Apply the room impulse responses to the latent signals
        for k in range(c.K):
            # Get the indices of the microphones for this node
            micIdx = np.arange(c.Mkc[k], c.Mkc[k + 1])
            # Apply the room impulse responses to the latent signals
            for ii in range(c.Qd):
                # ---------------------------------------
                if p.obsMat[k, ii] == 0:
                    continue  # Skip if the node does not observe the source
                # ---------------------------------------
                for jj, m in enumerate(micIdx):
                    tmp = sig.fftconvolve(
                        self.latentDesired[ii, smIdx[0]:smIdx[1]],
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

        # Store the STFT signals in the nodes
        for k in range(c.K):
            self.nodes[k].wd['s'][..., idxFrameBeg:idxFrameEnd] = s[c.Mkc[k]:c.Mkc[k + 1], ...]
            self.nodes[k].wd['n'][..., idxFrameBeg:idxFrameEnd] = n[c.Mkc[k]:c.Mkc[k + 1], ...]

    def define_layout(self):
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
            tmp = np.random.rand(3) *\
                (np.array(rd) - 2 * c.minDistFromWall) + c.minDistFromWall
            if c.onPlane:
                tmp[2] = zPlane
            while np.any(
                np.linalg.norm(tmp - np.array(nodesPos[:k, :]), axis=1) < c.nodeRadius * 2.5
            ):
                # If the node is too close to another node, generate a new position
                tmp = np.random.rand(3) *\
                    (np.array(rd) - 2 * c.minDistFromWall) + c.minDistFromWall
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
            attempt = np.random.rand(3) *\
                (np.array(rd) - 2 * c.minDistFromWall) + c.minDistFromWall
            if c.onPlane:
                attempt[2] = zPlane
            counter = 0
            while np.linalg.norm(attempt - nodesPos, axis=1).min() < c.minDistNodeSource:
                print(f"Desired source {ii + 1}/{c.Qd} too close to a node, generating a new position (trial #{counter+1})...", end='\r')
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
                print(f"Noise source {ii + 1}/{c.Qn} too close to a node or a desired source, generating a new position (trial #{counter+1})...", end='\r')
                # If the source is too close to a node, generate a new position
                attempt = np.random.rand(3) *\
                    (np.array(rd) - 2 * c.minDistFromWall) + c.minDistFromWall
                if c.onPlane:
                    attempt[2] = zPlane
                counter += 1
            # Store the position of the source
            noiseSourcesPos[ii, :] = attempt

        return nodesPos, sensorsPos, speechSourcesPos, noiseSourcesPos

    def define_static_scenario(self, room: pra.ShoeBox, omFixed=None) -> StaticScenarioParameters:
        c = self.cfg  # Configuration object

        # Define the layout of the acoustic scenario
        nodesPos, sensorsPos, speechSourcesPos, noiseSourcesPos =\
            self.define_layout()
        
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

    def get_observability_matrix(self, nodesPos=None, speechPos=None, noisePos=None) -> np.ndarray:
        """Compute the observability matrix."""
        c = self.cfg
        if c.observability == 'foss':
            # Full observability -- all nodes observe all sources
            obsMat = np.ones((c.K, c.Qd + c.Qn))
        elif c.observability == 'poss':
            # Partial observability -- nodes only observe certain sources
            if 1 or nodesPos is None or speechPos is None or noisePos is None:
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


def load_sound_file(file_path, desFs):
    """Load a sound file and resample it to the desired sampling frequency."""
    soundData, fsRead = sf.read(file_path, dtype='float32')
    if fsRead != desFs:
        # Resample the signal to the desired sampling frequency
        soundData = resample(soundData, fsRead, desFs)
    # Apply high-pass filter (get rid of potential low-frequency hum from low-quality dataset)
    # soundData = butter_highpass_filter(soundData, 0.01, 5)
    # Normalize the signal
    soundData /= np.amax(np.abs(soundData))  # Normalize
    soundData -= np.mean(soundData)  # Remove DC offset
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


def single_update_scm(RssPrev, RnnPrev, ssH, nnH, beta, vad=None):
    """Update the SCMs using the online estimation formula."""
    Rss = copy.deepcopy(RssPrev)  # by default, copy the previous SCM
    Rnn = copy.deepcopy(RnnPrev)  # by default, copy the previous SCM
    if vad is not None:
        raise NotImplementedError("VAD functionality is not correctly implemented.")
    # if 0:  # DEBUG: use this to test the VAD functionality
        if any(vad):
            Rss = beta * RssPrev + (1 - beta) * ssH
            # no update for noise SCM if VAD is False
        else:
            Rss = beta * RssPrev + (1 - beta) * ssH
            if RnnPrev is not None:
                # Rnn = beta * RnnPrev + (1 - beta) * yyH
                Rnn = beta * RnnPrev + (1 - beta) * nnH
    else:
        Rss = beta * RssPrev + (1 - beta) * ssH
        if RnnPrev is not None:
            Rnn = beta * RnnPrev + (1 - beta) * nnH
    return Rss, Rnn