# Contents of package:
# Classes and functions related to ....
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import pickle
from .base import *
import numpy as np
import networkx as nx
from tqdm import tqdm
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
        # Generate tree
        if c.graphDiameter is not None:
            graph = generate_tree_with_diameter(c.K, c.graphDiameter)
        else:
            graph = nx.complete_graph(c.K)
            for (u, v) in graph.edges():
                graph.edges[u, v]['weight'] = np.random.random()
            graph = nx.minimum_spanning_tree(graph)

        # Launch algorithms
        Ryy, Rss, Rnn, s, n = asc.setup()
        
        # Iterative variables for DANSE-like algorithms
        algDims = {
            'danse': c.Mk + c.Qd * (c.K - 1),
            'rsdanse': c.Mk + c.Qd * (c.K - 1),
            'tidanse': c.Mk + c.Qd,
        }
        iv = dict([(alg, {
            'tRyy': [
                1e-6 * c.randmat(
                    (c.nPosFreqs, algDims[alg], algDims[alg]) if c.domain == 'wola' else (algDims[alg], algDims[alg]),
                    makeComplex=True if c.domain != 'time' else False
                ) for _ in range(c.K)
            ],
            'tRnn': [
                1e-6 * c.randmat(
                    (c.nPosFreqs, algDims[alg], algDims[alg]) if c.domain == 'wola' else (algDims[alg], algDims[alg]),
                    makeComplex=True if c.domain != 'time' else False
                ) for _ in range(c.K)
            ],
            'Pk': [
                self.init_full((c.nPosFreqs, c.Mk, c.Qd), random=True)
                for _ in range(c.K)
            ],
            'WkkPrev': [
                self.init_full((c.nPosFreqs, c.Mk, c.Qd))
                for _ in range(c.K)
            ],
            'Wk': [
                self.init_full((c.nPosFreqs, algDims[alg], c.D), random=True)
                for _ in range(c.K)
            ],
            'u': 0,
            'gamma': np.ones(c.nPosFreqs),  # normalization factor for TI-DANSE
        }) for alg in c.algos if 'danse' in alg])  # iteration variable for DANSE algorithms

        if c.scmEstimation == 'online':
            W_netWide = [None for _ in range(c.nFrames)]
            for l in tqdm(range(c.nFrames), desc="Processing frames"):
                # Current frame information
                iv['frameIdx'] = l
                if c.domain == 'wola':
                    iv['frame_n'] = n[..., l].T
                    iv['frame_y'] = s[..., l].T + n[..., l].T
                else:
                    idxBeg = int(l * (c.frameLength * c.fs))
                    idxEnd = int((l + 1) * (c.frameLength * c.fs))
                    iv['frame_n'] = n[..., idxBeg:idxEnd].T
                    iv['frame_y'] = s[..., idxBeg:idxEnd].T + n[..., idxBeg:idxEnd].T
                # Launch algorithms for the current frame
                W_netWide[l], ivOut = self.launch(
                    Ryy[l], Rss[l], Rnn[l],
                    asc, graph,
                    ivIn=iv,
                    silent=True
                )
                # Feedback loop: update iterative variables for DANSE-like algorithms
                for alg in ivOut.keys():
                    for key, value in ivOut[alg].items():
                        iv[alg][key] = value
        else:
            W_netWide = self.launch(
                Ryy, Rss, Rnn,
                asc, graph,
                ivIn=iv
            )[0]

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

    def launch(
            self,
            RyyAll, RssAll, RnnAll,
            asc: AcousticScenario, G,
            ivIn=None,
            silent=False):
        """
        Launch algorithms.

        Parameters:
            RyyAll (dict[float, np.ndarray]): Full signal covariance matrix, for all beta values.
            RssAll (dict[float, np.ndarray]): Desired signal covariance matrix, for all beta values.
            RnnAll (dict[float, np.ndarray]): Noise signal covariance matrix, for all beta values.
            asc (AcousticScenario): Acoustic scenario object.
            G (nx.Graph): Graph representing the network topology.
            ivDANSE (dict[str, dict]): Iterative variables for each DANSE algorithm.
            silent (bool): If True, suppress output messages.
        """
        c = self.cfg
        W_netWide = dict([(alg, [
            self.init_full((c.nPosFreqs, c.M, c.D))
            for _ in range(c.K)
        ]) for alg in c.algos])  # Initialize node-specific filters dictionary
        ivOut = dict([(alg, None) for alg in c.algos if 'danse' in alg])
        for alg in c.algos:
            if not silent:
                print(f"Running algorithm: {alg}...")

            if c.scmEstimation == 'online':
                Ryy = RyyAll[c.beta[alg]]
                Rss = RssAll[c.beta[alg]]
                Rnn = RnnAll[c.beta[alg]]
            else:
                Ryy = RyyAll
                Rss = RssAll
                Rnn = RnnAll

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
                    # tRnn = reg(herm(Ck) @ Rnn @ Ck)
                    tRnn = herm(Ck) @ Rnn @ Ck
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
                            hEq[c.Mk * k: c.Mk * k + c.Q, :] = np.eye(c.Q)
                            Rhyqyktq = herm(Cqk[q]) @ Ryy @ hEq
                            Pk[q] = self.filtup(Rhyqhyq, Rss=Rhyqyktq)
                    # Compute estimation filter
                    tRyy = herm(Cqk[k]) @ Ryy @ Cqk[k]
                    # tRnn = reg(herm(Cqk[k]) @ Rnn @ Cqk[k])
                    tRnn = herm(Cqk[k]) @ Rnn @ Cqk[k]
                    W_netWide[alg][k] = Cqk[k] @ self.filtup(tRyy, tRnn, gevd=c.gevd, gevdRank=c.Qd)[..., :c.D]

            elif 'danse' in alg:
                # Extract the iterative variables
                Pk = ivIn[alg]['Pk']
                WkkPrev = ivIn[alg]['WkkPrev']
                Wk_Prev = ivIn[alg]['Wk']
                u = ivIn[alg]['u']
                tRyyPrev = ivIn[alg]['tRyy']
                tRnnPrev = ivIn[alg]['tRnn']
                onlineModeCriterion = True
                if c.scmEstimation == 'online':
                    frame_n = ivIn['frame_n']
                    frame_y = ivIn['frame_y']
                    onlineModeCriterion = ivIn['frameIdx'] % c.DANSEiterEveryXframes == 0
                gamma = ivIn[alg]['gamma']  # normalization factor for TI-DANSE

                # Wk = [[] for _ in range(c.K)]
                W_netWide[alg] = [[] for _ in range(c.K)]
                for i in range(c.maxDANSEiter):
                    if not silent:
                        print(f"Iteration {i + 1}/{c.maxDANSEiter} for {alg}...", end='\r')
                    if c.scmEstimation == 'online':
                        # Compute fused signals
                        zy, zn = [None for _ in range(c.K)], [None for _ in range(c.K)]
                        for k in range(c.K):
                            if c.domain == 'wola':
                                zy[k] = np.einsum(
                                    'ijk,ij->ik',
                                    Pk[k].conj(),
                                    frame_y[:, c.Mk * k:c.Mk * (k + 1)]
                                )
                                zn[k] = np.einsum(
                                    'ijk,ij->ik',
                                    Pk[k].conj(),
                                    frame_n[:, c.Mk * k:c.Mk * (k + 1)]
                                )
                            else:
                                # Time-domain-like processing
                                zy[k] = frame_y[:, c.Mk * k:c.Mk * (k + 1)] @ Pk[k].conj()
                                zn[k] = frame_n[:, c.Mk * k:c.Mk * (k + 1)] @ Pk[k].conj()
                        # Apply normalization factor for TI-DANSE
                        zy[k] *= np.conj(gamma)
                        zn[k] *= np.conj(gamma)

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
                        # Compute the SCMs
                        if c.scmEstimation == 'online':
                            # Build observation vector
                            if alg.startswith("tidanse"):
                                ty = np.concatenate([
                                    frame_y[:, c.Mk * k:c.Mk * (k + 1)],
                                    np.sum([zy[q] for q in range(c.K) if q != k], axis=0)
                                ], axis=1)
                                tn = np.concatenate([
                                    frame_n[:, c.Mk * k:c.Mk * (k + 1)],
                                    np.sum([zn[q] for q in range(c.K) if q != k], axis=0)
                                ], axis=1)
                            else:
                                ty = np.concatenate(
                                    [frame_y[:, c.Mk * k:c.Mk * (k + 1)]] +\
                                    [zy[q] for q in range(c.K) if q != k], axis=1
                                )
                                tn = np.concatenate(
                                    [frame_n[:, c.Mk * k:c.Mk * (k + 1)]] +\
                                    [zn[q] for q in range(c.K) if q != k], axis=1
                                )
                            if c.domain == 'wola':
                                yyH = np.einsum('ij,ik->ijk', ty, ty.conj())
                                nnH = np.einsum('ij,ik->ijk', tn, tn.conj())
                            else:
                                yyH = ty.T @ ty.conj()
                                nnH = tn.T @ tn.conj()
                            tRyy = c.beta[alg] * tRyyPrev[k] + (1 - c.beta[alg]) * yyH
                            tRnn = c.beta[alg] * tRnnPrev[k] + (1 - c.beta[alg]) * nnH
                            tRyyPrev[k] = tRyy
                            tRnnPrev[k] = tRnn
                        else:
                            tRyy = herm(Ck) @ Ryy @ Ck
                            tRnn = herm(Ck) @ Rnn @ Ck
                        
                        # Update the filters and fusion matrices
                        if (k == u or alg.startswith("rsdanse")) and onlineModeCriterion:
                            
                            # Compute the filter
                            tW = self.filtup(tRyy, tRnn, gevd=c.gevd, gevdRank=c.Qd)
                            if alg.startswith("rsdanse"):
                                # For rS-DANSE, we apply a relaxation
                                alpha = 1 / np.log10(i + 10)
                                tW[..., :c.Mk, :c.Qd] = (1 - alpha) * WkkPrev[k] +\
                                    alpha * tW[..., :c.Mk, :c.Qd]
                                WkkPrev[k] = tW[..., :c.Mk, :c.Qd]

                            if alg.startswith("tidanse"):
                                try:
                                    Pk[k] = tW[..., :c.Mk, :c.Qd] @\
                                        np.linalg.inv(tW[..., c.Mk:, :c.Qd])
                                except np.linalg.LinAlgError:
                                    if not silent:
                                        print("Matrix inversion failed, using pseudo-inverse instead.", end='\r')
                                    Pk[k] = tW[..., :c.Mk, :c.Qd] @\
                                        np.linalg.pinv(tW[..., c.Mk:, :c.Qd])
                            else:
                                Pk[k] = tW[..., :c.Mk, :c.Qd]

                            # Wk[k].append(tW[..., :c.D])
                            W_netWide[alg][k].append(Ck @ tW[..., :c.D])
                            Wk_Prev[k] = tW[..., :c.D]  # Store the last filter for the next iteration
                        else:
                            # Otherwise, we just store the previous filter
                            # if alg.startswith("tidanse"):
                            #     Nk = np.array([np.diag(
                            #         [1] * c.Mk + [gamma[kappa]] * c.Qd
                            #     ) for kappa in range(c.nPosFreqs)])
                            #     W_netWide[alg][k].append(
                            #         np.einsum(
                            #             'ijk->kij',
                            #             Nk,
                            #             W_NW_Prev[k]
                            #         )
                            #     )
                            # else:
                            W_netWide[alg][k].append(Ck @ Wk_Prev[k])
                    
                    # Update the updating node index for next iteration
                    if onlineModeCriterion:
                        u = (u + 1) % c.K
                
                # Store the iterative variables for the next frame
                ivOut[alg] = {
                    'Pk': Pk,
                    'WkkPrev': WkkPrev,
                    'W_NW': Wk_Prev,  # Last filter for each node
                    'u': u,
                    'tRyy': tRyyPrev,
                    'tRnn': tRnnPrev,
                }
            else:
                raise ValueError(f"Unknown algorithm: {alg}")
            
        return W_netWide, ivOut

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
