# Contents of package:
# Classes and functions related to ....
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import copy
import pickle
from .base import *
import numpy as np
import networkx as nx
from tqdm import tqdm
from .tree_utils import *
import scipy.linalg as sla
from dataclasses import dataclass
from .asc import AcousticScenario, single_update_scm
import matplotlib.pyplot as plt
from pyinstrument import Profiler


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
        if c.useVAD:
            vad, vadSTFT = asc.estimate_vad()

        # Iterative variables for DANSE-like algorithms
        algDims = {
            'danse': c.Mk + c.Qd * (c.K - 1),
            'rsdanse': c.Mk + c.Qd * (c.K - 1),
            'tidanse': c.Mk + c.Qd,
        }
        baseListDANSEscms = {
            alg: [
                1e-6 * c.randmat(
                    (c.nPosFreqs, algDims[alg], algDims[alg])
                    if c.domain == 'wola'
                    else (algDims[alg], algDims[alg]),
                    makeComplex=True if c.domain != 'time' else False
                ) for _ in range(c.K)
            ] for alg in c.algos if 'danse' in alg
        }
        iv = dict([(alg, {
            'tRyy': baseListDANSEscms[alg],
            'tRss': copy.deepcopy(baseListDANSEscms[alg]),
            'tRnn': copy.deepcopy(baseListDANSEscms[alg]),
            'Pk': [
                c.init_full((c.nPosFreqs, c.Mk, c.Qd), random=True)
                for _ in range(c.K)
            ],
            'WkkPrev_rS': [
                c.init_full((c.nPosFreqs, c.Mk, c.Qd))
                for _ in range(c.K)
            ],
            'Wk': [
                c.init_full((c.nPosFreqs, algDims[alg], algDims[alg]), random=True)
                for _ in range(c.K)
            ],
            'u': 0,
            'gamma': np.array([np.eye(c.Qd) for _ in range(c.nPosFreqs)]) if c.domain == 'wola' else np.eye(c.Qd),  # normalization factor for TI-DANSE
        }) for alg in c.algos if 'danse' in alg])  # iteration variable for DANSE algorithms

        if c.scmEstimation == 'online':
            W_netWide = []
            betas = list(set(c.beta.values()))

            Ryy = {
                beta: 1e-6 * c.randmat((c.nPosFreqs, c.M, c.M), makeComplex=True)
                for beta in betas
            }
            Rss = copy.deepcopy(Ryy)  # Initialize Rss with the same structure as Ryy
            Rnn = copy.deepcopy(Ryy)  # Initialize Rnn with the same structure as Ryy

            flagAlternating = np.any('dmwf' in alg for alg in c.algos) and c.dMWFalternating
            if flagAlternating:
                # For alternating dMWF, we compute SCMs using every other frame
                Ryy_est = copy.deepcopy(Ryy)
                Rss_est = copy.deepcopy(Ryy)
                Rnn_est = copy.deepcopy(Ryy)
                Ryy_dis = copy.deepcopy(Ryy)
                Rss_dis = copy.deepcopy(Ryy)
                Rnn_dis = copy.deepcopy(Ryy)
            else:
                Ryy_est, Rss_est, Rnn_est, Ryy_dis, Rss_dis, Rnn_dis = None, None, None, None, None, None  # default -- no alternating dMWF

            for l in tqdm(range(c.nFrames), desc=f"Processing frames ({len(c.algos)} algorithms)"):
                # New SCM estimates
                nnH = np.einsum('ji,ki->ijk', n[..., l], n[..., l].conj())
                if c.noCrossCorrelation:
                    # ssH
                    inner2 = np.einsum('ji,ki->ijk', s[..., l], s[..., l].conj())
                else:
                    # yyH
                    inner2 = np.einsum('ji,ki->ijk', (s + n)[..., l], (s + n)[..., l].conj())
                # Update the SCMs using the online estimation formula
                for beta in betas:
                    kwargs = {
                        'beta': beta,
                        'ssH': inner2,
                        'nnH': nnH,
                        'vad': vadSTFT[:, l] if c.useVAD else None
                    }
                    # Update the SCMs using the online estimation formula
                    if c.noCrossCorrelation:
                        Rss[beta], Rnn[beta] = single_update_scm(Rss[beta], Rnn[beta], **kwargs)
                    else:
                        Ryy[beta], Rnn[beta] = single_update_scm(Ryy[beta], Rnn[beta], **kwargs)
                    if l % 2 == 0 and flagAlternating:
                        # If alternating dMWF, update the dMWF estimation SCMs
                        # using the even frames only
                        if c.noCrossCorrelation:
                            Rss_est[beta], Rnn_est[beta] = single_update_scm(Rss_est[beta], Rnn_est[beta], **kwargs)
                        else:
                            Ryy_est[beta], Rnn_est[beta] = single_update_scm(Ryy_est[beta], Rnn_est[beta], **kwargs)
                    elif l % 2 == 1 and flagAlternating:
                        # If alternating dMWF, update the dMWF discovery SCMs
                        # using the odd frames only
                        if c.noCrossCorrelation:
                            Rss_dis[beta], Rnn_dis[beta] = single_update_scm(Rss_dis[beta], Rnn_dis[beta], **kwargs)
                        else:
                            Ryy_dis[beta], Rnn_dis[beta] = single_update_scm(Ryy_dis[beta], Rnn_dis[beta], **kwargs)

                    # Complete signal SCM
                    if c.noCrossCorrelation:
                        Ryy[beta] = Rss[beta] + Rnn[beta]
                        if flagAlternating:
                            Ryy_est[beta] = Rss_est[beta] + Rnn_est[beta]
                            Ryy_dis[beta] = Rss_dis[beta] + Rnn_dis[beta]
                    else:
                        Rss[beta] = None

                # Current frame information
                iv['frameIdx'] = l
                iv['vad'] = vadSTFT[:, l] if c.useVAD else None
                scenarioIdx = 0  # by default
                if c.domain == 'wola':
                    iv['frame_y'] = s[..., l].T + n[..., l].T
                    iv['frame_s'] = s[..., l].T
                    iv['frame_n'] = n[..., l].T
                    # Identify current acoustic scenario (for oQq and Qkq values in dMWF)
                    if c.dynamics == 'moving':
                        scenarioIdx = int((l * (c.nfft - c.nhop) / c.fs) // c.movingEvery)
                else:
                    idxBeg = int(l * (c.frameDuration * c.fs))
                    idxEnd = int((l + 1) * (c.frameDuration * c.fs))
                    iv['frame_y'] = s[..., idxBeg:idxEnd].T + n[..., idxBeg:idxEnd].T
                    iv['frame_s'] = s[..., idxBeg:idxEnd].T
                    iv['frame_n'] = n[..., idxBeg:idxEnd].T
                    # Identify current acoustic scenario (for oQq and Qkq values in dMWF)
                    if c.dynamics == 'moving':
                        scenarioIdx = int((l * c.frameDuration) // c.movingEvery)
                
                # Launch algorithms for the current frame
                # profile = Profiler()
                # profile.start()
                tmp, ivOut = self.launch(
                    Ryy, Rss, Rnn,
                    asc, graph,
                    ivIn=iv,
                    silent=True,
                    scenarioIdx=scenarioIdx,
                    Ryy_dMWF_estAll=Ryy_est,
                    Rnn_dMWF_estAll=Rnn_est,
                    Ryy_dMWF_disAll=Ryy_dis,
                )
                W_netWide.append(tmp)
                # profile.stop()
                # print(profile.output_text(unicode=True, color=True, show_all=True))

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
            silent=False,
            scenarioIdx=0,
            Ryy_dMWF_estAll=None,
            Rnn_dMWF_estAll=None,
            Ryy_dMWF_disAll=None,
        ):
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
            scenarioIdx (int): Index of the current scenario (for dynamic scenarios).
            Ryy_dMWF_estAll (dict[float, np.ndarray]): Full signal covariance matrix for dMWF estimation step with alternating discovery and estimation steps.
            Rnn_dMWF_estAll (dict[float, np.ndarray]): Noise signal covariance matrix for dMWF estimation step with alternating discovery and estimation steps.
            Ryy_dMWF_disAll (dict[float, np.ndarray]): Full signal covariance matrix for dMWF discovery step with alternating discovery and estimation steps.
        """
        c = self.cfg
        W_netWide = dict([(alg, [
            c.init_full((c.nPosFreqs, c.M, c.D))
            for _ in range(c.K)
        ]) for alg in c.algos])  # Initialize node-specific filters dictionary
        ivOut = dict([(alg, None) for alg in c.algos if 'danse' in alg])

        if Ryy_dMWF_estAll is None:
            Ryy_dMWF_estAll = RyyAll
        if Rnn_dMWF_estAll is None:
            Rnn_dMWF_estAll = RnnAll
        if Ryy_dMWF_disAll is None:
            Ryy_dMWF_disAll = RyyAll

        for alg in c.algos:
            if not silent:
                print(f"Running algorithm: {alg}...")

            if c.scmEstimation == 'online':
                Ryy = RyyAll[c.beta[alg]]
                Rss = RssAll[c.beta[alg]]
                Rnn = RnnAll[c.beta[alg]]
                Ryy_dMWF_est = Ryy_dMWF_estAll[c.beta[alg]]
                Rnn_dMWF_est = Rnn_dMWF_estAll[c.beta[alg]]
                Ryy_dMWF_dis = Ryy_dMWF_disAll[c.beta[alg]]
            else:
                Ryy = RyyAll
                Rss = RssAll
                Rnn = RnnAll
                Ryy_dMWF_est = Ryy_dMWF_estAll
                Rnn_dMWF_est = Rnn_dMWF_estAll
                Ryy_dMWF_dis = Ryy_dMWF_disAll

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
                scn = asc.scenarios[scenarioIdx]  # Current acoustic scenario
                # Neighbor-specific fusion matrices
                Pk = [None for _ in range(c.K)]
                for q in range(c.K):
                    Ryqyq = Ryy_dMWF_dis[..., c.Mk * q:c.Mk * (q + 1), c.Mk * q:c.Mk * (q + 1)]
                    Ryqrhoq = c.init_full((c.nPosFreqs, c.Mk, scn.oQq[q]))
                    for p in range(c.K):
                        if p == q:
                            continue
                        # Eqps = c.init_full((c.Mk, scn.oQq[q]), selection_matrix=True)
                        # # Eqps[:scn.oQq[q], :scn.oQq[q]] = np.eye(scn.oQq[q])
                        # Eqps[:scn.Qkq[q, p], :scn.Qkq[q, p]] = np.eye(scn.Qkq[q, p])
                        # Eqps[scn.Qkq[q, p]:, scn.Qkq[q, p]:] = np.ones((c.Mk - scn.Qkq[q, p], scn.oQq[q] - scn.Qkq[q, p]))
                        Ryqrhoq += Ryy_dMWF_dis[..., c.Mk * q:c.Mk * (q + 1), c.Mk * p:c.Mk * (p + 1)] @ scn.Eqps[q][p]
                    Pk[q] = self.filtup(Ryqyq, Rss=Ryqrhoq)
                # Estimation filters
                for k in range(c.K):
                    # ty = C^H.y
                    QkqNeighs = np.delete(scn.oQq, k)  # Remove k
                    Ck = c.init_full((c.nPosFreqs, c.M, c.Mk + int(np.sum(QkqNeighs))))
                    Ck[..., c.Mk * k:c.Mk * (k + 1), :c.Mk] = np.eye(c.Mk)
                    idxNei = 0
                    for q in range(c.K):
                        if q != k:
                            idxBeg = c.Mk + int(np.sum(QkqNeighs[:idxNei]))
                            idxEnd = idxBeg + scn.oQq[q]
                            Ck[..., c.Mk * q:c.Mk * (q + 1), idxBeg:idxEnd] = Pk[q]
                            idxNei += 1
                    # Compute the filters
                    tRyy = herm(Ck) @ Ryy_dMWF_est @ Ck
                    tRnn = herm(Ck) @ Rnn_dMWF_est @ Ck
                    tW = self.filtup(tRyy, tRnn, gevd=c.gevd, gevdRank=c.Qd)[..., :c.D]
                    W_netWide[alg][k] = Ck @ tW

            elif alg == "tidmwf":
                for k in range(c.K):
                    _, upstreamNeighs = get_upstream_nodes(G, k)
                    Cqk = [None for _ in range(c.K)]
                    for q in range(c.K):
                        dim = c.Mk + c.Q * len(upstreamNeighs[q])
                        Cqk[q] = c.init_full((c.nPosFreqs, c.M, dim))
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
                            hEq = c.init_full((c.M, c.Q), selection_matrix=True)
                            hEq[c.Mk * k: c.Mk * k + c.Q, :] = np.eye(c.Q)
                            Rhyqyktq = herm(Cqk[q]) @ Ryy @ hEq
                            Pk[q] = self.filtup(Rhyqhyq, Rss=Rhyqyktq)
                    # Compute estimation filter
                    tRyy = herm(Cqk[k]) @ Ryy @ Cqk[k]
                    tRnn = herm(Cqk[k]) @ Rnn @ Cqk[k]
                    W_netWide[alg][k] = Cqk[k] @ self.filtup(tRyy, tRnn, gevd=c.gevd, gevdRank=c.Qd)[..., :c.D]

            elif 'danse' in alg:
                # Extract the iterative variables
                Pk = ivIn[alg]['Pk']
                WkkPrev_rS = ivIn[alg]['WkkPrev_rS']
                Wk = ivIn[alg]['Wk']
                u = ivIn[alg]['u']
                tRyyPrev = ivIn[alg]['tRyy']
                tRssPrev = ivIn[alg]['tRss']
                tRnnPrev = ivIn[alg]['tRnn']
                onlineModeCriterion = True
                if c.scmEstimation == 'online':
                    frame_y = ivIn['frame_y']
                    frame_s = ivIn['frame_s']
                    frame_n = ivIn['frame_n']
                    onlineModeCriterion = ivIn['frameIdx'] % c.DANSEiterEveryXframes == 0
                gamma = ivIn[alg]['gamma']  # normalization factor for TI-DANSE

                W_netWide[alg] = [[] for _ in range(c.K)]
                for i in range(c.maxDANSEiter):
                    if not silent:
                        print(f"Iteration {i + 1}/{c.maxDANSEiter} for {alg}...", end='\r')

                    # For TI-DANSE, take normalization factor into account
                    Nk = np.array([
                        sla.block_diag(*(np.eye(c.Mk), gamma[kappa]))
                        for kappa in range(c.nPosFreqs)
                    ]) if c.domain == 'wola' else sla.block_diag(*(np.eye(c.Mk), gamma))

                    if c.scmEstimation == 'online':
                        # Compute fused signals
                        zy, zs, zn = [None for _ in range(c.K)], [None for _ in range(c.K)], [None for _ in range(c.K)]
                        for k in range(c.K):
                            if c.domain == 'wola':
                                zy[k] = np.einsum(
                                    'ijk,ij->ik',
                                    Pk[k].conj(),
                                    frame_y[:, c.Mk * k:c.Mk * (k + 1)]
                                )
                                zs[k] = np.einsum(
                                    'ijk,ij->ik',
                                    Pk[k].conj(),
                                    frame_s[:, c.Mk * k:c.Mk * (k + 1)]
                                )
                                zn[k] = np.einsum(
                                    'ijk,ij->ik',
                                    Pk[k].conj(),
                                    frame_n[:, c.Mk * k:c.Mk * (k + 1)]
                                )
                            else:
                                # Time-domain-like processing
                                zy[k] = frame_y[:, c.Mk * k:c.Mk * (k + 1)] @ Pk[k].conj()
                                zs[k] = frame_s[:, c.Mk * k:c.Mk * (k + 1)] @ Pk[k].conj()
                                zn[k] = frame_n[:, c.Mk * k:c.Mk * (k + 1)] @ Pk[k].conj()
                            
                            if alg.startswith("tidanse"):
                                # Apply normalization factor for TI-DANSE
                                if c.domain == 'wola':
                                    zy[k] = np.einsum('ijk,ik->ij', herm(gamma), zy[k])
                                    zs[k] = np.einsum('ijk,ik->ij', herm(gamma), zs[k])
                                    zn[k] = np.einsum('ijk,ik->ij', herm(gamma), zn[k])
                                else:
                                    zy[k] = np.einsum('ij,kj->ki', herm(gamma), zy[k])
                                    zs[k] = np.einsum('ij,kj->ki', herm(gamma), zs[k])
                                    zn[k] = np.einsum('ij,kj->ki', herm(gamma), zn[k])

                    for k in range(c.K):
                        # Compute C-matrix
                        if alg.startswith("tidanse"):
                            Ck = c.init_full((c.nPosFreqs, c.M, c.Mk + c.Qd))
                            Ck[..., c.Mk * k:c.Mk * (k + 1), :c.Mk] = np.eye(c.Mk)
                            for q in range(c.K):
                                if q != k:
                                    Ck[..., c.Mk * q:c.Mk * (q + 1), c.Mk:] = Pk[q] @ gamma
                        else:
                            Ck = c.init_full((c.nPosFreqs, c.M, c.Mk + c.Qd * (c.K - 1)))
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
                                ts = np.concatenate([
                                    frame_s[:, c.Mk * k:c.Mk * (k + 1)],
                                    np.sum([zs[q] for q in range(c.K) if q != k], axis=0)
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
                                ts = np.concatenate(
                                    [frame_s[:, c.Mk * k:c.Mk * (k + 1)]] +\
                                    [zs[q] for q in range(c.K) if q != k], axis=1
                                )
                                tn = np.concatenate(
                                    [frame_n[:, c.Mk * k:c.Mk * (k + 1)]] +\
                                    [zn[q] for q in range(c.K) if q != k], axis=1
                                )
                            if c.domain == 'wola':
                                yyH = np.einsum('ij,ik->ijk', ty, ty.conj())
                                ssH = np.einsum('ij,ik->ijk', ts, ts.conj())
                                nnH = np.einsum('ij,ik->ijk', tn, tn.conj())
                            else:
                                yyH = ty.T @ ty.conj()
                                ssH = ts.T @ ts.conj()
                                nnH = tn.T @ tn.conj()
                            
                            if alg.startswith("tidanse"):
                                tRyyPrev[k] = herm(Nk) @ tRyyPrev[k] @ Nk
                                tRssPrev[k] = herm(Nk) @ tRssPrev[k] @ Nk
                                tRnnPrev[k] = herm(Nk) @ tRnnPrev[k] @ Nk

                            # Update the SCMs
                            if c.noCrossCorrelation:
                                tRss, tRnn = single_update_scm(
                                    tRssPrev[k], tRnnPrev[k],
                                    ssH, nnH, c.beta[alg],
                                    vad=ivIn['vad']
                                )
                                tRyy = tRss + tRnn
                            else:
                                tRyy, tRnn = single_update_scm(
                                    tRyyPrev[k], tRnnPrev[k],
                                    yyH, nnH, c.beta[alg],
                                    vad=ivIn['vad']
                                )
                                tRss = None
                        else:
                            tRnn = herm(Ck) @ Rnn @ Ck
                            if c.noCrossCorrelation:
                                tRss = herm(Ck) @ Rss @ Ck
                                tRyy = tRss + tRnn
                            else:
                                tRyy = herm(Ck) @ Ryy @ Ck
                                tRss = None
                        
                        tRyyPrev[k] = tRyy
                        tRssPrev[k] = tRss
                        tRnnPrev[k] = tRnn

                        # Update the filters
                        if (k == u or alg.startswith("rsdanse")) and onlineModeCriterion:
                            # Compute the filter
                            Wk[k] = self.filtup(tRyy, tRnn, gevd=c.gevd, gevdRank=c.Qd)

                            if alg.startswith("rsdanse"):
                                # For rS-DANSE, we apply a relaxation
                                alpha = 1 / np.log10(i + 10)
                                Wk[k][..., :c.Mk, :c.Qd] = (1 - alpha) * WkkPrev_rS[k] +\
                                    alpha * Wk[k][..., :c.Mk, :c.Qd]
                                WkkPrev_rS[k] = Wk[k][..., :c.Mk, :c.Qd]
                        else:
                            # No update for this node
                            if alg.startswith("tidanse"): # and c.scmEstimation == 'online':
                                # For TI-DANSE, apply the normalization factor
                                Wk[k] = np.linalg.inv(Nk) @ Wk[k]  # loaded from previous iteration/frame
                        
                        # Compute the fusion matrix Pk
                        if alg.startswith("tidanse"):
                            Pk[k] = Wk[k][..., :c.Mk, :c.Qd] @\
                                np.linalg.inv(Wk[k][..., c.Mk:, :c.Qd])
                            if k == 0:
                                print(f'\nNorm of Pk[{k}]: {np.linalg.norm(Pk[k])}')
                        else:
                            Pk[k] = Wk[k][..., :c.Mk, :c.Qd]

                        # Store the network-wide filter for this iteration/frame
                        W_netWide[alg][k].append(Ck @ Wk[k][..., :c.D])
                        
                    # Update the normalization factor for TI-DANSE
                    if alg.startswith("tidanse"): # and c.scmEstimation == 'online':
                        # Update anyway, always, at the reference node
                        r = c.refNodeForTInorm
                        tWr = self.filtup(tRyyPrev[r], tRnnPrev[r], gevd=c.gevd, gevdRank=c.Qd)
                        gamma = tWr[..., c.Mk:, :c.Qd]
                
                    # Update the updating node index for next iteration
                    if onlineModeCriterion:
                        u = (u + 1) % c.K
                
                # Store the iterative variables for the next frame
                ivOut[alg] = {
                    'Pk': Pk,
                    'WkkPrev_rS': WkkPrev_rS,
                    'Wk': Wk,  # Last filter for each node
                    'u': u,
                    'tRyy': tRyyPrev,
                    'tRss': tRssPrev,
                    'tRnn': tRnnPrev,
                    'gamma': gamma,  # normalization factor for TI-DANSE
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
                # tmp = np.linalg.solve(c.mu * Ryy + (1 - c.mu) * Rss, Rss)
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