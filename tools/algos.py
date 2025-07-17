# Contents of package:
# Classes and functions related to ....
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import pickle
from .base import *
import numpy as np
import networkx as nx
from .tree_utils import *
import scipy.linalg as sla
from dataclasses import dataclass
from .asc import AcousticScenario
import matplotlib.pyplot as plt


@dataclass
class Run:
    cfg: Parameters

    def go(self):
        # Generate scenario
        c = self.cfg
        asc = AcousticScenario(cfg=c)
        Ryy, Rss, Rnn, s, n = asc.setup()
        
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

        # Export results
        self.export_results(W_netWide, s, n)

        return 0

    def export_results(self, W_netWide, s, n):
        """Export results to a file."""
        c = self.cfg
        results = {
            'W_netWide': W_netWide,
            's': s,
            'n': n,
            'cfg': c,
        }
        with open(c.outputFilePath, 'wb') as f:
            pickle.dump(results, f)
        # Export cfg as .txt file
        with open(c.outputFilePath.replace('.pkl', '.txt'), 'w') as f:
            f.write(str(c))
        print(f"Results exported to {c.outputFilePath}")

    def launch(self, Ryy, Rss, Rnn, asc: AcousticScenario, G):
        """Launch algorithms."""
        c = self.cfg
        W_netWide = dict([(alg, [
            self.init_full((c.nPosFreqs, c.M, c.D))
            for _ in range(c.K)
        ]) for alg in c.algos])  # Initialize node-specific filters dictionary
        for alg in c.algos:
            print(f"Running algorithm: {alg}...")
            if alg == 'unprocessed':
                for k in range(c.K):
                    W_netWide[alg][k][..., c.Mk * k:c.Mk * k + c.D, :] = np.eye(c.D)
            elif alg == "centralized":
                Wcentr = self.filtup(Ryy, Rnn, gevd=c.gevd, gevdRank=c.Qd)
                W_netWide[alg] = [
                    Wcentr[..., c.Mk * k:c.Mk * k + c.D] for k in range(c.K)
                ]
            elif alg == "local":
                for k in range(c.K):
                    Rykyk = Ryy[..., c.Mk * k:c.Mk * (k + 1), c.Mk * k:c.Mk * (k + 1)]
                    Rnknk = Rnn[..., c.Mk * k:c.Mk * (k + 1), c.Mk * k:c.Mk * (k + 1)]
                    tmp = self.filtup(Rykyk, Rnknk, gevd=c.gevd, gevdRank=c.Qd)
                    W_netWide[alg][k][..., c.Mk * k:c.Mk * (k + 1), :] = tmp[..., :c.D]
            elif alg == "dmwf":
                # Neighbor-specific fusion matrices
                Pk = [None for _ in range(c.K)]
                for q in range(c.K):
                    Ryqyq = Ryy[..., c.Mk * q:c.Mk * (q + 1), c.Mk * q:c.Mk * (q + 1)]
                    Ryqrhoq = self.init_full((c.nPosFreqs, c.Mk, asc.oQq[q]))
                    for p in range(c.K):
                        if p == q:
                            continue
                        Eqps = self.init_full((c.Mk, asc.oQq[q]), selection_matrix=True)
                        Eqps[:asc.Qkq[q, p], :asc.Qkq[q, p]] = np.eye(asc.Qkq[q, p])
                        # Pad with ones
                        Eqps[asc.Qkq[q, p]:, asc.Qkq[q, p]:] = np.ones((c.Mk - asc.Qkq[q, p], asc.oQq[q] - asc.Qkq[q, p]))
                        Ryqrhoq += Ryy[..., c.Mk * q:c.Mk * (q + 1), c.Mk * p:c.Mk * (p + 1)] @ Eqps
                    Pk[q] = self.filtup(Ryqyq, Rss=Ryqrhoq)
                # Estimation filters
                for k in range(c.K):
                    # ty = C^H.y
                    QkqNeighs = np.delete(asc.oQq, k)  # Remove k
                    Ck = self.init_full((c.nPosFreqs, c.M, c.Mk + int(np.sum(QkqNeighs))))
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
                    tRnn = reg(herm(Ck) @ Rnn @ Ck)
                    tW = self.filtup(tRyy, tRnn, gevd=c.gevd, gevdRank=c.Qd)[..., :c.D]
                    W_netWide[alg][k] = Ck @ tW
                    pass

            elif alg == "tidmwf":
                for k in range(c.K):
                    _, upstreamNeighs = get_upstream_nodes(G, k)
                    _, downstreamNeigh = get_downstream_nodes(G, k)
                    Cqk = [None for _ in range(c.K)]
                    for q in range(c.K):
                        dim = c.Mk + c.Q * len(upstreamNeighs[q])
                        Cqk[q] = self.init_full((c.nPosFreqs, c.M, dim))
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
                            hEq = self.init_full((c.M, c.Q), selection_matrix=True)
                            # hEq[
                            #     c.Mk * downstreamNeigh[q]:\
                            #     c.Mk * downstreamNeigh[q] + c.Q, :
                            # ] = np.eye(c.Q)
                            hEq[c.Mk * k: c.Mk * k + c.Q, :] = np.eye(c.Q)
                            Rhyqyktq = herm(Cqk[q]) @ Ryy @ hEq
                            Pk[q] = self.filtup(Rhyqhyq, Rss=Rhyqyktq)
                    # Compute estimation filter
                    tRyy = herm(Cqk[k]) @ Ryy @ Cqk[k]
                    tRnn = reg(herm(Cqk[k]) @ Rnn @ Cqk[k])
                    W_netWide[alg][k] = Cqk[k] @ self.filtup(tRyy, tRnn, gevd=c.gevd, gevdRank=c.Qd)[..., :c.D]

                pass

            elif 'danse' in alg:
                W_netWide[alg] = [[] for _ in range(c.K)]
                # Initialize the fusion matrices
                Pk = [
                    self.init_full((c.nPosFreqs, c.Mk, c.Qd), random=True)
                    for _ in range(c.K)
                ]
                WkkPrev = [
                    self.init_full((c.nPosFreqs, c.Mk, c.Qd))
                    for _ in range(c.K)
                ]
                u = 0  # updating node index
                for i in range(c.maxDANSEiter):
                    print(f"Iteration {i + 1}/{c.maxDANSEiter} for {alg}...", end='\r')
                    for k in range(c.K):
                        if alg.startswith("tidanse"):
                            Ck = self.init_full((c.nPosFreqs, c.M, c.Mk + c.Qd))
                            Ck[..., c.Mk * k:c.Mk * (k + 1), :c.Mk] = np.eye(c.Mk)
                            for q in range(c.K):
                                if q != k:
                                    Ck[..., c.Mk * q:c.Mk * (q + 1), c.Mk:] = Pk[q]
                        else:
                            Ck = self.init_full((c.nPosFreqs, c.M, c.Mk + c.Qd * (c.K - 1)))
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
                        tRnn = herm(Ck) @ Rnn @ Ck
                        tW = self.filtup(tRyy, tRnn, gevd=c.gevd, gevdRank=c.Qd)
                        if alg.startswith("rsdanse"):
                            # For rS-DANSE, we apply a relaxation
                            alpha = 1 / np.log10(i + 10)
                            tW[..., :c.Mk, :c.Qd] = (1 - alpha) * WkkPrev[k] +\
                                alpha * tW[..., :c.Mk, :c.Qd]
                            WkkPrev[k] = tW[..., :c.Mk, :c.Qd]
                        W_netWide[alg][k].append(Ck @ tW[..., :c.D])
                        # Update the fusion matrices
                        if k == u or alg.startswith("rsdanse"):
                            if alg.startswith("tidanse"):
                                try:
                                    Pk[k] = tW[..., :c.Mk, :c.Qd] @\
                                        np.linalg.inv(tW[..., c.Mk:, :c.Qd])
                                except np.linalg.LinAlgError:
                                    print("Matrix inversion failed, using pseudo-inverse instead.", end='\r')
                                    Pk[k] = tW[..., :c.Mk, :c.Qd] @\
                                        np.linalg.pinv(tW[..., c.Mk:, :c.Qd])
                            else:
                                Pk[k] = tW[..., :c.Mk, :c.Qd]
                    u = (u + 1) % c.K  # Update the node index for next iteration
            else:
                raise ValueError(f"Unknown algorithm: {alg}")
            
        return W_netWide

    def filtup(self, Ryy, Rnn=None, Rss=None, gevd=False, gevdRank=1):
        """Filter up the SCMs."""
        c = self.cfg

        if Rnn is not None and Rss is None:
            Rss = Ryy - Rnn

        finalSize = Rss.shape[-1]
        if finalSize < Ryy.shape[-1]:
            # To apply the SDW addition, we need to pad Ryy (then discard the extra columns later)
            if c.domain == 'wola':
                Rss = np.pad(Rss, ((0, 0), (0, 0), (0, Ryy.shape[-1] - finalSize)), mode='constant')
            elif 'time' in c.domain:
                Rss = np.pad(Rss, ((0, 0), (0, Ryy.shape[-1] - finalSize)), mode='constant')

        def _regular_mwf(Ryy, Rss):
            try:
                tmp = np.linalg.inv(c.mu * Ryy + (1 - c.mu) * Rss) @ Rss
            except np.linalg.LinAlgError:
                print("Matrix inversion failed, using pseudo-inverse instead.", end='\r')
                tmp = np.linalg.pinv(c.mu * Ryy + (1 - c.mu) * Rss) @ Rss
            return tmp

        if gevd:
            sigFull = np.zeros((Ryy.shape[0], Ryy.shape[1]), dtype=complex)
            Xfull = np.zeros((Ryy.shape[0], Ryy.shape[1], Ryy.shape[1]), dtype=complex)
            for f in range(Ryy.shape[0]):
                try:
                    sigma, Xmat = sla.eigh(Ryy[f, ...], Rnn[f, ...])
                except np.linalg.LinAlgError:
                    print("GEVD failed, using pseudo-inverse instead.", end='\r')
                    return _regular_mwf(Ryy, Rss)[..., :finalSize]

                idx = np.flip(np.argsort(sigma))
                sigFull[f, :] = sigma[idx]
                Xfull[f, ...] = Xmat[:, idx]
            # Inverse-hermitian of `Xmat`
            Qmat = np.linalg.inv(Xfull.conj().transpose(0, 2, 1))
            # GEVLs tensor - low-rank approximation is done here
            Dmat = np.zeros_like(Ryy, dtype=complex)
            for r in range(gevdRank):
                Dmat[:, r, r] = np.squeeze(1 - 1 / sigFull[:, r])
            # Compute filters
            w = Xfull @ Dmat @ Qmat.conj().transpose(0, 2, 1)
            return w[..., :finalSize]
        else:
            tmp = _regular_mwf(Ryy, Rss)
        return tmp[..., :finalSize]
    
    def init_full(self, shape, value=0, random=False, selection_matrix=False):
        """Initialize a full matrix."""
        c = self.cfg
        if random:
            if 'time' in c.domain and not selection_matrix:
                shape = (shape[1], shape[2])  # get rid of the frequency dimension
            return c.randmat(shape)
        if c.domain == 'wola':
            return np.full(shape, value, dtype=complex)
        elif 'time' in c.domain:
            if not selection_matrix:
                shape = (shape[1], shape[2])  # get rid of the frequency dimension
            return np.full(shape, value, dtype=complex if c.domain == 'time_complex' else float)

def flatten_list(l):
    """Flatten a list of lists."""
    return [item for sublist in l for item in sublist]


def reg(B_batch, epsilon=1e-8):
    """Check if a matrix is positive definite."""
    # if not np.all(B == herm(B)):
    #     B_reg = (B + herm(B)) / 2
    #     assert np.all(B_reg == herm(B_reg)), "Matrix is not Hermitian after regularization."
    # else:
    #     B_reg = B

    # if np.any(np.linalg.eigvals(B_reg) <= 0) or np.iscomplex(np.linalg.eigvals(B_reg)).any():
    #     B_reg += 1e-8 * np.eye(B_reg.shape[-1])
    #     assert np.all(np.linalg.eigvals(B_reg) > 0), "Matrix is not positive definite after regularization."

    M, N, _ = B_batch.shape
    B_psd = np.zeros_like(B_batch, dtype=np.complex128)

    for m in range(M):
        Bm = (B_batch[m] + B_batch[m].conj().T) / 2  # Symmetrize

        evals, evecs = np.linalg.eigh(Bm)

        evals_clipped = np.clip(evals, epsilon, None)

        Bm_psd = (evecs @ np.diag(evals_clipped)) @ evecs.conj().T

        B_psd[m] = Bm_psd

    return B_psd
