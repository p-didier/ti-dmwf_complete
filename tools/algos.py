# Contents of package:
# Classes and functions related to ....
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import time
import numpy as np
import networkx as nx
from numba import njit
from .tree_utils import *
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pyinstrument import Profiler
from .asc import AcousticScenario
from .base import Parameters, randmat

@dataclass
class Run:
    cfg: Parameters

    def go(self):
        # Generate scenario
        c = self.cfg
        asc = AcousticScenario(cfg=c)
        Ryy, Rss, Rnn, s, n, v = asc.setup()
        y = s + n + v  # Observed signal (centralized)
        d = np.array([s[c.Mk * k:c.Mk * k + c.D, ...] for k in range(c.K)])  # target signals
        
        # Generate tree
        if c.graphDiameter is not None:
            graph = generate_tree_with_diameter(c.K, c.graphDiameter)
        else:
            graph = nx.complete_graph(c.K)
            for (u, v) in graph.edges():
                graph.edges[u, v]['weight'] = np.random.random()
            graph = nx.minimum_spanning_tree(graph)

        # Launch algorithms
        W_netWide = self.launch(Ryy, Rss, Rnn, asc, graph)

        # Compute metrics
        print("\nComputing metrics...")
        t0 = time.time()
        metrics = self.get_metrics(W_netWide, y, d)
        print(f"\nMetrics computed in {time.time() - t0:.2f} seconds.")

        # Post-process results
        self.plot_metrics(metrics)

    def plot_metrics(self, metrics):
        c = self.cfg
        fig, axes = plt.subplots(1, len(c.metricsToCompute))
        fig.set_size_inches(8.5, 3.5)
        for ii, m in enumerate(c.metricsToCompute):
            ax = axes[ii] if len(c.metricsToCompute) > 1 else axes
            if any('danse' in alg for alg in c.algos):
                ax.set_yscale('log')
                for ii, alg in enumerate(metrics[m].keys()):
                    if m == 'msew' and alg in ['centralized', 'local','unprocessed']:
                        continue
                    if 'danse' in alg:
                        if metrics[m][alg] is not None:
                            data = np.mean(metrics[m][alg], axis=0)
                            ax.plot(data, label=alg)
                        else:
                            print(f"No {m} data for {alg}, skipping.")
                    else:
                        # Non-iterative algorithms
                        ax.axhline(y=np.mean(metrics[m][alg]), linestyle='--', label=alg, color=f'C{ii}')
                ax.set_xlim(0, c.maxDANSEiter)
            else:
                pass  # TODO
            ax.legend()
            ax.set_title(m)
        fig.suptitle(f'{c.observability}, {c.scmEstimation}')
        fig.tight_layout()
        plt.show()

    def get_metrics(self, W_netWide, y, d):
        c = self.cfg

        metrics = dict([(metric, dict([
            (alg, [None for _ in range(c.K)]) for alg in c.algos + ['unprocessed']
        ])) for metric in c.metricsToCompute])


        def _process_msed(dk=None, dhatk=None):
            """Process MSEd metric."""
            if dk.shape[0] == c.nfft // 2 + 1 and dhatk.shape[0] == c.nfft // 2 + 1:
                # Compute ISTFT -- back to time domain
                dkTD = c.get_istft(dk.transpose(1, 0, 2))
                dhatkTD = c.get_istft(dhatk.transpose(1, 0, 2))
                return np.mean(np.abs(dkTD - dhatkTD) ** 2)
            return np.mean(np.abs(dk - dhatk) ** 2)
        
        def _process(Wk, hWk=None, dk=None, dhatk=None,):
            metrics_curr = dict([(metric, None) for metric in c.metricsToCompute])
            if 'msew' in c.metricsToCompute:
                metrics_curr['msew'] = np.mean(np.abs(Wk - hWk) ** 2)
            if 'msed' in c.metricsToCompute:
                metrics_curr['msed'] = _process_msed(dk, dhatk)
            return metrics_curr
        
        msedOverFrames = 50 # Number of frames to average MSEd over

        yc = y[..., :msedOverFrames].transpose(1, 0, 2)  # Centralized signal for MSEd computation
        for k in range(c.K):
            hWk = W_netWide['centralized'][k]
            dk = d[k, ..., :msedOverFrames].transpose(1, 0, 2)  # Desired signal for node k
            if 'msed' in c.metricsToCompute:
                # Compute unprocessed MSE
                metrics['msed']['unprocessed'][k] = _process_msed(
                    dk=dk, dhatk=yc[:, c.Mk * k:c.Mk * k + c.D, :]
                )
            for alg in c.algos:
                if alg == 'tidmwf':
                    pass
                print(f"Computing metrics for {alg}, node {k}...", end='\r')
                if isinstance(W_netWide[alg][k], list):
                    for ii in range(len(W_netWide[alg][k])):
                        dhatk = herm(W_netWide[alg][k][ii]) @ yc
                        metric_curr = _process(W_netWide[alg][k][ii], hWk=hWk, dk=dk, dhatk=dhatk)
                        for m in c.metricsToCompute:
                            if metrics[m][alg][k] is None:
                                metrics[m][alg][k] = []
                            metrics[m][alg][k].append(metric_curr[m])
                else:
                    dhatk = herm(W_netWide[alg][k]) @ yc
                    metric_curr = _process(W_netWide[alg][k], hWk=hWk, dk=dk, dhatk=dhatk)
                    for m in c.metricsToCompute:
                        metrics[m][alg][k] = metric_curr[m]
                pass

        return metrics

    def launch(self, Ryy, Rss, Rnn, asc: AcousticScenario, G):
        """Launch algorithms."""
        c = self.cfg
        W_netWide = dict([(alg, [
            np.zeros((c.nPosFreqs, c.M, c.D), dtype=complex)
            for _ in range(c.K)
        ]) for alg in c.algos])  # Initialize node-specific filters dictionary
        for alg in c.algos:
            print(f"Running algorithm: {alg}...")
            if alg == "centralized":
                Wcentr = self.filtup(Ryy, Rss)
                W_netWide[alg] = [
                    Wcentr[..., c.Mk * k:c.Mk * k + c.D] for k in range(c.K)
                ]
            elif alg == "local":
                for k in range(c.K):
                    Rykyk = Ryy[..., c.Mk * k:c.Mk * (k + 1), c.Mk * k:c.Mk * (k + 1)]
                    Rsksk = Rss[..., c.Mk * k:c.Mk * (k + 1), c.Mk * k:c.Mk * (k + 1)]
                    tmp = self.filtup(Rykyk, Rsksk)
                    W_netWide[alg][k][:, c.Mk * k:c.Mk * (k + 1), :] = tmp[..., :c.D]
            elif alg == "dmwf":
                # Neighbor-specific fusion matrices
                Pk = [None for _ in range(c.K)]
                for q in range(c.K):
                    Ryqyq = Ryy[..., c.Mk * q:c.Mk * (q + 1), c.Mk * q:c.Mk * (q + 1)]
                    Rgqgqu = np.zeros((c.nPosFreqs, c.Mk, asc.oQq[q]), dtype=complex)
                    for p in range(c.K):
                        if p == q:
                            continue
                        Eqps = np.zeros((c.Mk, asc.oQq[q]), dtype=complex)
                        Eqps[:asc.Qkq[q, p], :asc.Qkq[q, p]] = np.eye(asc.Qkq[q, p])
                        Eqps[asc.Qkq[q, p]:, asc.Qkq[q, p]:] = np.ones((c.Mk- asc.Qkq[q, p], asc.oQq[q] - asc.Qkq[q, p]))
                        Rgqgqu += Ryy[..., c.Mk * q:c.Mk * (q + 1), c.Mk * p:c.Mk * (p + 1)] @ Eqps
                    Pk[q] = self.filtup(Ryqyq, Rgqgqu)
                # Estimation filters
                for k in range(c.K):
                    # ty = C^H.y
                    QkqNeighs = np.delete(asc.oQq, k)  # Remove k
                    Ck = np.zeros((c.nPosFreqs, c.M, c.Mk + int(np.sum(QkqNeighs))), dtype=complex)
                    Ck[..., c.Mk * k:c.Mk * (k + 1), :c.Mk] = np.eye(c.Mk)
                    idxNei = 0
                    for q in range(c.K):
                        if q != k:
                            idxBeg = c.Mk + int(np.sum(QkqNeighs[:idxNei]))
                            idxEnd = idxBeg + asc.oQq[q]
                            Ck[..., c.Mk * q:c.Mk * (q + 1), idxBeg:idxEnd] = Pk[q]
                            idxNei += 1
                    # Compute the filters
                    tRyy = herm(Ck) @ Ryy @ Ck
                    tRss = herm(Ck) @ Rss @ Ck
                    tW = self.filtup(tRyy, tRss)[..., :c.D]
                    W_netWide[alg][k] = Ck @ tW

            elif alg == "tidmwf":
                for k in range(c.K):
                    if k == 2:
                        pass
                    upstreamNodes, upstreamNeighs = get_upstream_nodes(G, k)
                    downstreamNodes, downstreamNeigh = get_downstream_nodes(G, k)
                    Cqk = [None for _ in range(c.K)]
                    for q in range(c.K):
                        dim = c.Mk + c.Q * len(upstreamNeighs[q])
                        Cqk[q] = np.zeros((c.nPosFreqs, c.M, dim), dtype=complex)
                        Cqk[q][..., c.Mk * q:c.Mk * (q + 1), :c.Mk] = np.eye(c.Mk)
                    # Compute fusion matrices
                    Pk = [None for _ in range(c.K)]
                    for q in flatten_list(tree_levels(G, k)):
                        for ii, n in enumerate(upstreamNeighs[q]):
                            idxBeg = c.Mk + ii * c.Q
                            idxEnd = idxBeg + c.Q
                            Cqk[q][..., idxBeg:idxEnd] = Cqk[n] @ Pk[n]
                        if q != k:
                            # Compute Pk
                            Rhyqhyq = herm(Cqk[q]) @ Ryy @ Cqk[q]
                            hEq = np.zeros((c.M, c.Q))
                            hEq[
                                c.Mk * downstreamNeigh[q]:\
                                c.Mk * downstreamNeigh[q] + c.Q, :
                            ] = np.eye(c.Q)
                            Rhyqyktq = herm(Cqk[q]) @ Ryy @ hEq
                            # Pk[q] = np.linalg.inv(Rhyqhyq) @ Rhyqyktq
                            Pk[q] = self.filtup(Rhyqhyq, Rhyqyktq)
                    # Compute estimation filter
                    tRyy = herm(Cqk[k]) @ Ryy @ Cqk[k]
                    tRss = herm(Cqk[k]) @ Rss @ Cqk[k]
                    W_netWide[alg][k] = Cqk[k] @ self.filtup(tRyy, tRss)[..., :c.D]

            elif 'danse' in alg:
                W_netWide[alg] = [[] for _ in range(c.K)]
                # Initialize the fusion matrices
                Pk = [randmat((c.nPosFreqs, c.Mk, c.Qd)) for _ in range(c.K)]
                WkkPrev = [
                    np.zeros((c.nPosFreqs, c.Mk, c.Qd), dtype=complex)
                    for _ in range(c.K)
                ]
                u = 0  # updating node index
                for i in range(c.maxDANSEiter):
                    print(f"Iteration {i + 1}/{c.maxDANSEiter} for {alg}...", end='\r')
                    for k in range(c.K):
                        if alg.startswith("tidanse"):
                            Ck = np.zeros((c.nPosFreqs, c.M, c.Mk + c.Qd), dtype=complex)
                            Ck[..., c.Mk * k:c.Mk * (k + 1), :c.Mk] = np.eye(c.Mk)
                            for q in range(c.K):
                                if q != k:
                                    Ck[..., c.Mk * q:c.Mk * (q + 1), c.Mk:] = Pk[q]
                        else:
                            Ck = np.zeros((c.nPosFreqs, c.M, c.Mk + c.Qd * (c.K - 1)), dtype=complex)
                            Ck[..., c.Mk * k:c.Mk * (k + 1), :c.Mk] = np.eye(c.Mk)
                            idxNei = 0
                            for q in range(c.K):
                                if q != k:
                                    idxBeg = c.Mk + idxNei * c.Qd
                                    idxEnd = idxBeg + c.Qd
                                    Ck[..., c.Mk * q:c.Mk * (q + 1), idxBeg:idxEnd] = Pk[q]
                                    idxNei += 1
                        # Compute the filters
                        tRyy = herm(Ck) @ Ryy @ Ck
                        tRss = herm(Ck) @ Rss @ Ck
                        tW = self.filtup(tRyy, tRss)
                        if alg.startswith("rsdanse"):
                            alpha = 1 / np.log10(i + 10)
                            tW[..., :c.Mk, :c.Qd] = (1 - alpha) * WkkPrev[k] +\
                                alpha * tW[..., :c.Mk, :c.Qd]
                            WkkPrev[k] = tW[..., :c.Mk, :c.Qd]
                        if k == u or alg.startswith("rsdanse"):
                            if alg.startswith("tidanse"):
                                Pk[k] = tW[..., :c.Mk, :c.Qd] @\
                                    np.linalg.inv(tW[..., c.Mk:, :c.Qd])
                            else:
                                Pk[k] = tW[..., :c.Mk, :c.Qd]
                        W_netWide[alg][k].append(Ck @ tW[..., :c.D])
                    u = (u + 1) % c.K  # Update the node index for next iteration
            else:
                raise ValueError(f"Unknown algorithm: {alg}")
            
        return W_netWide

    def filtup(self, Ryy, Rss):
        """Filter up the SCMs."""
        c = self.cfg
        finalSize = Rss.shape[-1]
        if finalSize < Ryy.shape[-1]:
            Rss = np.pad(Rss, ((0, 0), (0, 0), (0, Ryy.shape[-1] - finalSize)), mode='constant')
        tmp = np.linalg.inv(c.mu * Ryy + (1 - c.mu) * Rss) @ Rss
        return tmp[..., :finalSize]

def flatten_list(l):
    """Flatten a list of lists."""
    return [item for sublist in l for item in sublist]


def herm(x: np.ndarray) -> np.ndarray:
    """Hermitian transpose."""
    if x.ndim == 2:
        return x.conj().T
    elif x.ndim == 3:
        return x.conj().transpose(0, 2, 1)
