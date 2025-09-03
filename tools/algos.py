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
import matplotlib.pyplot as plt
from pyinstrument import Profiler
from dataclasses import dataclass
from humanfriendly import format_timespan
from .asc import AcousticScenario, single_update_scm, single_update_scm_inplace


@dataclass
class Run:
    cfg: Parameters

    def go(self):
        # Generate scenario
        tMaster = time.time()
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
        out = asc.setup()
        Ryy, Rss, Rnn = out[0], out[1], out[2]
        if c.domain == 'wola':
            sigStack = out[3]
            s, n = sigStack['s'], sigStack['n']
        else:
            s, n = out[3], out[4]
            sigStack = {
                's': s,
                'n': n
            }
        if c.useVAD:
            vad, vadSTFT = asc.estimate_vad()

        # Iterative variables for DANSE-like algorithms
        refScn = asc.scenarios[0]  # Reference scenario for DANSE-like algorithms (ASSUMING NO CHANGING OBSERVABILITY)
        algDims = {
            'danse': [
                c.Mk[k] + np.sum([refScn.Qdk[q] for q in range(c.K) if q != k])
                for k in range(c.K)
            ],
            'rsdanse': [
                c.Mk[k] + np.sum([refScn.Qdk[q] for q in range(c.K) if q != k])
                for k in range(c.K)
            ],
            'tidanse': [
                c.Mk[k] + c.Qd for k in range(c.K)
            ],
        }
        baseListDANSEscms = {
            alg: [
                c.scmInitScaling * c.randmat_hermposdef(
                    (c.nPosFreqs, algDims[alg][k], algDims[alg][k])
                    if c.domain == 'wola'
                    else (algDims[alg][k], algDims[alg][k]),
                    makeComplex=True if c.domain != 'time' else False
                ) for k in range(c.K)
            ] for alg in c.algos if 'danse' in alg
        }
        iv = dict([(alg, {
            'tRyy': baseListDANSEscms[alg],
            'tRss': copy.deepcopy(baseListDANSEscms[alg]),
            'tRnn': copy.deepcopy(baseListDANSEscms[alg]),
            'nUpdates_tRyy': 0,
            'nUpdates_tRnn': 0,
            'Pk': [
                c.init_full((c.nPosFreqs, c.Mk[k], refScn.Qdk[k]), random=True)
                for k in range(c.K)
            ],
            'WkkPrev_rS': [
                c.init_full((c.nPosFreqs, c.Mk[k], refScn.Qdk[k]), random=True)
                for k in range(c.K)
            ],
            'Wk': [
                c.init_full((c.nPosFreqs, algDims[alg][k], algDims[alg][k]), random=True)
                for k in range(c.K)
            ],
            'u': 0,
            'i': 0,
            'gamma': np.array([np.eye(c.Qd) for _ in range(c.nPosFreqs)]) if c.domain == 'wola' else np.eye(c.Qd),  # normalization factor for TI-DANSE
        }) for alg in c.algos if 'danse' in alg])  # iteration variable for DANSE algorithms


        if c.scmEstimation == 'online':
            # Saving iteration index per frame per DANSE-like algo, for post-processing
            iSaved = dict([(alg, np.zeros(c.nFrames)) for alg in c.algos if 'danse' in alg])

            W_netWide = []

            if c.scmHeadStart is not None:
                outtt = asc.compute_scms(force=c.scmHeadStart)
                Rss, Rnn = outtt[1], outtt[2]
                Rss += c.scmHeadStartNoiseAmount * np.amax(np.abs(Rss)) *\
                    c.randmat_hermposdef(Rss.shape, makeComplex=True)
                Rnn += c.scmHeadStartNoiseAmount * np.amax(np.abs(Rnn)) *\
                    c.randmat_hermposdef(Rnn.shape, makeComplex=True)
                Ryy = Rss + Rnn
                Ryy = np.ascontiguousarray(Ryy, dtype=c.mydtype)
                Rss = np.ascontiguousarray(Rss, dtype=c.mydtype)
                Rnn = np.ascontiguousarray(Rnn, dtype=c.mydtype)
            else:
                Ryy = c.scmInitScaling * c.randmat_hermposdef((c.nPosFreqs, c.M, c.M), makeComplex=True)
                Rss = copy.deepcopy(Ryy)  # Initialize Rss with the same structure as Ryy
                Rnn = copy.deepcopy(Ryy)  # Initialize Rnn with the same structure as Ryy

            # For alternating dMWF, we compute SCMs using every other frame
            flagdMWF_alternating = np.any(['dmwf' in alg for alg in c.algos]) and c.dMWFalternating
            Ryy_est = copy.deepcopy(Ryy)
            Rss_est = copy.deepcopy(Rss)
            Rnn_est = copy.deepcopy(Rnn)
            Ryy_dis = copy.deepcopy(Ryy)
            Rss_dis = copy.deepcopy(Rss)
            Rnn_dis = copy.deepcopy(Rnn)

            for l in tqdm(range(c.nFrames), desc=f"Processing frames ({len(c.algos)} algorithms)"):
                sl = s[..., l]  # current frame of desired signal
                nl = n[..., l]  # current frame of noise
                yl = sl + nl

                # profiler = Profiler()
                # profiler.start()
                if c.algos == ['unprocessed']:
                    pass
                else:
                    # New SCM estimates
                    # nnH = np.einsum('ji,ki->ijk', nl, nl.conj())
                    nnH = np.einsum('ji,ki->ijk', yl, np.conjugate(yl))  # yyH
                    if c.noCrossCorrelation:
                        inner2 = np.einsum('ji,ki->ijk', sl, np.conjugate(sl))  # ssH
                    else: 
                        inner2 = np.einsum('ji,ki->ijk', yl, np.conjugate(yl))  # yyH
                    if flagdMWF_alternating:
                        slprev = s[..., l - 1]  # last frame of desired signal
                        nlprev = n[..., l - 1]  # last frame of noise
                        ylprev = slprev + nlprev
                        # Prepare previous chunk of data for alternating dMWF
                        # nnHprev = np.einsum('ji,ki->ijk', n[..., l - 1], n[..., l - 1].conj())
                        nnHprev = np.einsum('ji,ki->ijk', ylprev, np.conjugate(ylprev))  # yyH
                        if c.noCrossCorrelation:
                            inner2_prev = np.einsum('ji,ki->ijk', slprev, np.conjugate(slprev))  # ssH
                        else: 
                            inner2_prev = np.einsum('ji,ki->ijk', ylprev, np.conjugate(ylprev))  # yyH

                    # Update the SCMs using the online estimation formula
                    kwargs = {
                        'beta': c.beta,
                        'ssH': inner2,
                        'nnH': nnH,
                        'vad': vadSTFT[:, l] if c.useVAD else None,
                        # 'forceRyyUp': True if l < 107 else False  # DEBUGING, TEMPORARY!!!!
                    }
                    if flagdMWF_alternating:
                        kwargs_prev = copy.deepcopy(kwargs)
                        kwargs_prev['nnH'] = nnHprev
                        kwargs_prev['ssH'] = inner2_prev
                    
                    # Update the SCMs using the online estimation formula
                    if c.noCrossCorrelation:
                        Rss, Rnn = single_update_scm(Rss, Rnn, **kwargs)
                    else:
                        Ryy, Rnn = single_update_scm(Ryy, Rnn, **kwargs)
                    
                    if l % 2 == 0 and flagdMWF_alternating:
                        # If alternating dMWF, update the dMWF estimation SCMs
                        # using the even frames only, and update the dMWF
                        # discovery SCMs using the previous chunk of data.
                        if c.noCrossCorrelation:
                            Rss_est, Rnn_est = single_update_scm(Rss_est, Rnn_est, **kwargs)
                            Rss_dis, Rnn_dis = single_update_scm(Rss_dis, Rnn_dis, **kwargs_prev)
                        else:
                            Ryy_est, Rnn_est = single_update_scm(Ryy_est, Rnn_est, **kwargs)
                            Ryy_dis, Rnn_dis = single_update_scm(Ryy_dis, Rnn_dis, **kwargs_prev)
                    
                    elif l % 2 == 1 and flagdMWF_alternating:
                        # If alternating dMWF, update the dMWF discovery SCMs
                        # using the odd frames only, and update the dMWF
                        # estimation SCMs using the previous chunk of data.
                        if c.noCrossCorrelation:
                            Rss_dis, Rnn_dis = single_update_scm(Rss_dis, Rnn_dis, **kwargs)
                            Rss_est, Rnn_est = single_update_scm(Rss_est, Rnn_est, **kwargs_prev)
                        else:
                            Ryy_dis, Rnn_dis = single_update_scm(Ryy_dis, Rnn_dis, **kwargs)
                            Ryy_est, Rnn_est = single_update_scm(Ryy_est, Rnn_est, **kwargs_prev)

                    # Complete signal SCM
                    if c.noCrossCorrelation:
                        Ryy = Rss + Rnn
                        if flagdMWF_alternating:
                            Ryy_est = Rss_est + Rnn_est
                            Ryy_dis = Rss_dis + Rnn_dis
                    else:
                        Rss = None
                # profiler.stop()
                # print(profiler.output_text(unicode=True, color=True, show_all=True))
                pass

                # Current frame information
                iv['frameIdx'] = l
                iv['vad'] = vadSTFT[:, l] if c.useVAD else None
                scenarioIdx = 0  # by default
                if c.domain == 'wola':
                    iv['frame_y'] = sl.T + nl.T
                    iv['frame_s'] = sl.T
                    iv['frame_n'] = nl.T
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
                scenarioIdx = np.amin([scenarioIdx, len(asc.scenarios) - 1])
                
                # Launch algorithms for the current frame
                # profile = Profiler()
                # profile.start()
                tmp, ivOut = self.launch(
                    Ryy, Rss, Rnn,
                    asc, graph,
                    ivIn=iv,
                    silent=True,
                    scenarioIdx=scenarioIdx,
                    Ryy_dMWF_est=Ryy_est,
                    Rnn_dMWF_est=Rnn_est,
                    Ryy_dMWF_dis=Ryy_dis,
                )
                W_netWide.append(tmp)
                # profile.stop()
                # print(profile.output_text(unicode=True, color=True, show_all=True))
                pass

                # Feedback loop: update iterative variables for DANSE-like algorithms
                for alg in ivOut.keys():
                    iSaved[alg][l] = ivOut[alg]['i']  # save iteration index at current frame
                    for key, value in ivOut[alg].items():
                        iv[alg][key] = value
        else:
            W_netWide = self.launch(
                Ryy, Rss, Rnn,
                asc, graph,
                ivIn=iv
            )[0]
            iSaved = None  # placeholder

        # Export results
        self.export_results(W_netWide, sigStack, asc, iSaved)

        print(f"\nTotal time taken for this run: {format_timespan(time.time() - tMaster)}")

        return 0

    def export_results(self, W_netWide, sigStack, asc: AcousticScenario, iSaved):
        """Export results to a file."""
        c = self.cfg
        if c.scmEstimation == 'online' and c.desSigType == 'speech' and c.singleLine is None:
            # Wideband speech enhancement simulation: don't compute MSE_W and
            # don't export W_netWide. Instead, export the estimated desired
            # signal, the filtered s and filtered n, all in the time-domain.
            # Export the filtered signals _per source_ for metrics computation.
            shatk = [
                dict([(alg, [None for _ in range(c.Qd)]) for alg in c.algos])
                for _ in range(c.K)
            ]
            nhatk = [
                dict([(alg, [None for _ in range(c.Qn)]) for alg in c.algos])
                for _ in range(c.K)
            ]
            for alg in c.algos:
                for k in range(c.K):
                    print(f"Computing time-domain estimates for algorithm {alg}, node {k}...", end='\r')
                    # Compile coefficients
                    wCurr = np.array([
                        w[alg][k][0] if isinstance(w[alg][k], list) else w[alg][k]
                        for w in W_netWide
                    ])
                    for ii in range(c.Qd):
                        shatkSTFT = np.einsum(
                            'ijkl,kji->lji',
                            wCurr.conj(),
                            sigStack['sIndiv'][ii, ...]
                        )
                        shatk[k][alg][ii] = c.get_istft(shatkSTFT)
                    for ii in range(c.Qn):
                        nhatkSTFT = np.einsum(
                            'ijkl,kji->lji',
                            wCurr.conj(),
                            sigStack['nIndiv'][ii, ...]
                        )
                        nhatk[k][alg][ii] = c.get_istft(nhatkSTFT)

            results = {
                'd': [
                    c.get_istft(np.array([
                        sigStack['sIndiv'][ii, c.Mkc[k]:c.Mkc[k] + c.D, ...]
                        for k in range(c.K)
                    ]))
                    for ii in range(c.Qd)
                ],
                'shatk': shatk,
                'nhatk': nhatk,
                'asc': asc,  # (contains `Parameters` cfg)
            }
        else:
            results = {
                'W_netWide': W_netWide,
                's': sigStack['s'],
                'n': sigStack['n'],
                'asc': asc,  # (contains `Parameters` cfg)
            }
        # Saving iter index
        results['iSaved'] = iSaved

        with open(c.outputFilePath, 'wb') as f:
            pickle.dump(results, f)
        # Export cfg as .txt file
        with open(c.outputFilePath.replace('.pkl', '.txt'), 'w') as f:
            f.write(str(c))
        print(f"Results exported to {c.outputFilePath}")

    def launch(
            self,
            Ryy, Rss, Rnn,
            asc: AcousticScenario, G,
            ivIn=None,
            silent=False,
            scenarioIdx=0,
            Ryy_dMWF_est=None,
            Rnn_dMWF_est=None,
            Ryy_dMWF_dis=None,
        ):
        """
        Launch algorithms.

        Parameters:
            Ryy (np.ndarray): Full signal covariance matrix.
            Rss (np.ndarray): Desired signal covariance matrix.
            Rnn (np.ndarray): Noise signal covariance matrix.
            asc (AcousticScenario): Acoustic scenario object.
            G (nx.Graph): Graph representing the network topology.
            ivDANSE (dict[str, dict]): Iterative variables for each DANSE algorithm.
            silent (bool): If True, suppress output messages.
            scenarioIdx (int): Index of the current scenario (for dynamic scenarios).
            Ryy_dMWF_est (np.ndarray): Full signal covariance matrix for dMWF estimation step with alternating discovery and estimation steps.
            Rnn_dMWF_est (np.ndarray): Noise signal covariance matrix for dMWF estimation step with alternating discovery and estimation steps.
            Ryy_dMWF_dis (np.ndarray): Full signal covariance matrix for dMWF discovery step with alternating discovery and estimation steps.
        """
        c = self.cfg
        W_netWide = dict([(alg, [
            c.init_full((c.nPosFreqs, c.M, c.D))
            for _ in range(c.K)
        ]) for alg in c.algos])  # Initialize node-specific filters dictionary
        ivOut = dict([(alg, None) for alg in c.algos if 'danse' in alg])

        scn = asc.scenarios[scenarioIdx]  # Current acoustic scenario

        for alg in c.algos:
            if not silent:
                print(f"Running algorithm: {alg}...")

            if alg == 'unprocessed':
                for k in range(c.K):
                    W_netWide[alg][k][..., c.Mkc[k]:c.Mkc[k] + c.D, :] = np.eye(c.D)
            elif alg == "centralized":
                Wcentr = self.filtup(Ryy, Rnn, gevd=c.gevd, gevdRank=c.Qd)
                W_netWide[alg] = [
                    Wcentr[..., c.Mkc[k]:c.Mkc[k] + c.D] for k in range(c.K)
                ]
            elif alg == "local":
                for k in range(c.K):
                    Rykyk = Ryy[..., c.Mkc[k]:c.Mkc[k + 1], c.Mkc[k]:c.Mkc[k + 1]]
                    Rnknk = Rnn[..., c.Mkc[k]:c.Mkc[k + 1], c.Mkc[k]:c.Mkc[k + 1]]
                    tmp = self.filtup(Rykyk, Rnknk, gevd=c.gevd, gevdRank=c.Qd)
                    W_netWide[alg][k][..., c.Mkc[k]:c.Mkc[k + 1], :] = tmp[..., :c.D]
            elif alg == "dmwf":
                # Neighbor-specific fusion matrices
                Pk = [None for _ in range(c.K)]
                for q in range(c.K):
                    Ryqyq = Ryy_dMWF_dis[..., c.Mkc[q]:c.Mkc[q + 1], c.Mkc[q]:c.Mkc[q + 1]]
                    Ryqrhoq = c.init_full((c.nPosFreqs, c.Mk[q], scn.oQq[q]))
                    for p in range(c.K):
                        if p == q:
                            continue
                        Ryqrhoq += Ryy_dMWF_dis[
                            ...,
                            c.Mkc[q]:c.Mkc[q + 1],
                            c.Mkc[p]:c.Mkc[p + 1]
                        ] @ scn.Eqps[q][p]
                    Pk[q] = self.filtup(Ryqyq, Rss=Ryqrhoq)
                
                # Estimation filters
                for k in range(c.K):
                    # profiler = Profiler()
                    # profiler.start()
                    # Compute C to get ty = C^H.y
                    QkqNeighs = np.delete(scn.oQq, k)  # Remove k
                    # # For efficiency: preallocate Ck
                    # if "Ck" not in locals() or Ck.shape[2] != (c.Mk[k] + int(np.sum(QkqNeighs))):
                    #     Ck = np.zeros((c.nPosFreqs, c.M, c.Mk[k] + int(np.sum(QkqNeighs))), dtype=c.mydtype)
                    # else:
                    #     Ck.fill(0)   # reset instead of reallocating
                    Ck = c.init_full((c.nPosFreqs, c.M, c.Mk[k] + int(np.sum(QkqNeighs))))
                    Ck[..., c.Mkc[k]:c.Mkc[k + 1], :c.Mk[k]] = np.eye(c.Mk[k])
                    idxNei = 0
                    for q in range(c.K):
                        if q != k:
                            idxBeg = c.Mk[k] + int(np.sum(QkqNeighs[:idxNei]))
                            idxEnd = idxBeg + scn.oQq[q]
                            Ck[..., c.Mkc[q]:c.Mkc[q + 1], idxBeg:idxEnd] = Pk[q]
                            idxNei += 1
                    # Compute the filters
                    tRyy = matmul_CRC(Ck, Ryy_dMWF_est)
                    tRnn = matmul_CRC(Ck, Rnn_dMWF_est)
                    # tRyy = herm(Ck) @ Ryy_dMWF_est @ Ck
                    # tRnn = herm(Ck) @ Rnn_dMWF_est @ Ck
                    tW = self.filtup(tRyy, tRnn, gevd=c.gevd, gevdRank=c.Qd)[..., :c.D]
                    W_netWide[alg][k] = Ck @ tW
                    # profiler.stop()
                    # print(profiler.output_text(unicode=True, color=True, show_all=True))
                    pass


            elif alg == "tidmwf":
                # NB: alternating steps not implemented yet.
                for k in range(c.K):
                    _, upstreamNeighs = get_upstream_nodes(G, k)
                    Cqk = [None for _ in range(c.K)]
                    for q in range(c.K):
                        dim = c.Mk[q] + c.Q * len(upstreamNeighs[q])
                        Cqk[q] = c.init_full((c.nPosFreqs, c.M, dim))
                        Cqk[q][..., c.Mkc[q]:c.Mkc[q + 1], :c.Mk[q]] = np.eye(c.Mk[q])
                    # Compute fusion matrices
                    Pk = [None for _ in range(c.K)]
                    for q in flatten_list(tree_levels(G, k)):
                        for ii, n in enumerate(upstreamNeighs[q]):
                            idxBeg = c.Mk[q] + ii * c.Q
                            idxEnd = idxBeg + c.Q
                            Cqk[q][..., idxBeg:idxEnd] = Cqk[n] @ Pk[n]
                        if q != k:
                            # Compute Pk
                            Rhyqhyq = herm(Cqk[q]) @ Ryy @ Cqk[q]
                            hEq = c.init_full((c.M, c.Q), selection_matrix=True)
                            hEq[c.Mkc[k]:c.Mkc[k] + c.Q, :] = np.eye(c.Q)
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
                nUpdates_tRyy = ivIn[alg]['nUpdates_tRyy']
                nUpdates_tRnn = ivIn[alg]['nUpdates_tRnn']
                onlineModeCriterion = True
                if c.scmEstimation == 'online':
                    frame_y = ivIn['frame_y']
                    frame_s = ivIn['frame_s']
                    frame_n = ivIn['frame_n']
                    iEff = ivIn[alg]['i']  # effective iteration index
                    if not c.useVAD or c.ignoreVAD_DANSEiterEveryXframes:
                        if c.DANSEiterEveryXframes == -1:
                            c.DANSEiterEveryXframes = 1
                        onlineModeCriterion = ivIn['frameIdx'] % c.DANSEiterEveryXframes == 0
                gamma = ivIn[alg]['gamma']  # normalization factor for TI-DANSE

                W_netWide[alg] = [[] for _ in range(c.K)]
                for i in range(c.maxDANSEiter):
                    if not silent:
                        print(f"Iteration {i + 1}/{c.maxDANSEiter} for {alg}...", end='\r')
                    if c.scmEstimation != 'online':
                        iEff = i  # effective iteration index for batch mode

                    # For TI-DANSE, take normalization factor into account
                    Nk = [
                            np.array([
                            sla.block_diag(*(np.eye(c.Mk[k]), gamma[kappa]))
                            for kappa in range(c.nPosFreqs)
                        ]) if c.domain == 'wola' else sla.block_diag(*(np.eye(c.Mk[k]), gamma))
                        for k in range(c.K)
                    ]

                    if c.scmEstimation == 'online':
                        # Compute fused signals
                        zy, zs, zn = [None for _ in range(c.K)], [None for _ in range(c.K)], [None for _ in range(c.K)]
                        for k in range(c.K):
                            if c.domain == 'wola':
                                zy[k] = np.einsum(
                                    'ijk,ij->ik',
                                    Pk[k].conj(),
                                    frame_y[:, c.Mkc[k]:c.Mkc[k + 1]]
                                )
                                zs[k] = np.einsum(
                                    'ijk,ij->ik',
                                    Pk[k].conj(),
                                    frame_s[:, c.Mkc[k]:c.Mkc[k + 1]]
                                )
                                zn[k] = np.einsum(
                                    'ijk,ij->ik',
                                    Pk[k].conj(),
                                    frame_n[:, c.Mkc[k]:c.Mkc[k + 1]]
                                )
                            else:
                                # Time-domain-like processing
                                zy[k] = frame_y[:, c.Mkc[k]:c.Mkc[k + 1]] @ Pk[k].conj()
                                zs[k] = frame_s[:, c.Mkc[k]:c.Mkc[k + 1]] @ Pk[k].conj()
                                zn[k] = frame_n[:, c.Mkc[k]:c.Mkc[k + 1]] @ Pk[k].conj()
                            
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
                            Ck = c.init_full((c.nPosFreqs, c.M, c.Mk[k] + c.Qd))
                            Ck[..., c.Mkc[k]:c.Mkc[k + 1], :c.Mk[k]] = np.eye(c.Mk[k])
                            for q in range(c.K):
                                if q != k:
                                    Ck[..., c.Mkc[q]:c.Mkc[q + 1], c.Mk[k]:] = Pk[q] @ gamma
                        else:
                            Qdks = [scn.Qdk[q] for q in range(c.K) if q != k]
                            Ck = c.init_full((c.nPosFreqs, c.M, c.Mk[k] + np.sum(Qdks)))
                            Ck[..., c.Mkc[k]:c.Mkc[k + 1], :c.Mk[k]] = np.eye(c.Mk[k])
                            idxNei = 0
                            for q in range(c.K):
                                if q != k:
                                    idxBeg = int(c.Mk[k] + np.sum(Qdks[:idxNei]))
                                    idxEnd = idxBeg + Qdks[idxNei]
                                    Ck[..., c.Mkc[q]:c.Mkc[q + 1], idxBeg:idxEnd] = Pk[q]
                                    idxNei += 1
                        # Compute the SCMs
                        if c.scmEstimation == 'online':
                            # Build observation vector
                            if alg.startswith("tidanse"):
                                ty = np.concatenate([
                                    frame_y[:, c.Mkc[k]:c.Mkc[k + 1]],
                                    np.sum([zy[q] for q in range(c.K) if q != k], axis=0)
                                ], axis=1)
                                ts = np.concatenate([
                                    frame_s[:, c.Mkc[k]:c.Mkc[k + 1]],
                                    np.sum([zs[q] for q in range(c.K) if q != k], axis=0)
                                ], axis=1)
                                tn = np.concatenate([
                                    frame_n[:, c.Mkc[k]:c.Mkc[k + 1]],
                                    np.sum([zn[q] for q in range(c.K) if q != k], axis=0)
                                ], axis=1)
                            else:
                                ty = np.concatenate(
                                    [frame_y[:, c.Mkc[k]:c.Mkc[k + 1]]] +\
                                    [zy[q] for q in range(c.K) if q != k], axis=1
                                )
                                ts = np.concatenate(
                                    [frame_s[:, c.Mkc[k]:c.Mkc[k + 1]]] +\
                                    [zs[q] for q in range(c.K) if q != k], axis=1
                                )
                                tn = np.concatenate(
                                    [frame_n[:, c.Mkc[k]:c.Mkc[k + 1]]] +\
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
                                tRyyPrev[k] = herm(Nk[k]) @ tRyyPrev[k] @ Nk[k]
                                tRssPrev[k] = herm(Nk[k]) @ tRssPrev[k] @ Nk[k]
                                tRnnPrev[k] = herm(Nk[k]) @ tRnnPrev[k] @ Nk[k]

                            # Update the SCMs
                            # Counting based on node 0 (/!\ implicitly assumes
                            # frame-wise VAD is the same for all nodes)
                            if c.useVAD and k == 0:
                                # Only iterate DANSE if both SCMs have been
                                # updated enough times
                                if any(ivIn['vad']):
                                    # Speech+noise frame, Ryy is updated, Rnn not
                                    nUpdates_tRyy += 1
                                else:
                                    # Noise-only frame, Rnn is updated, Ryy not
                                    nUpdates_tRnn += 1
                                if not c.ignoreVAD_DANSEiterEveryXframes:
                                    if (nUpdates_tRyy >= c.DANSEiterEveryXframes and\
                                        nUpdates_tRnn >= c.DANSEiterEveryXframes) or\
                                        c.DANSEiterEveryXframes == -1:
                                        # Both Ryy and Rnn have been updated, we can proceed
                                        nUpdates_tRyy = 0
                                        nUpdates_tRnn = 0
                                        onlineModeCriterion = True  # new iteration
                                    else:
                                        onlineModeCriterion = False  # no new iteration yet

                            if c.noCrossCorrelation:
                                tRss, tRnn = single_update_scm(
                                    tRssPrev[k], tRnnPrev[k],
                                    ssH=ssH, nnH=nnH, beta=c.beta,
                                    vad=ivIn['vad']
                                )
                                tRyy = tRss + tRnn
                            else:
                                tRyy, tRnn = single_update_scm(
                                    tRyyPrev[k], tRnnPrev[k],
                                    ssH=yyH, nnH=nnH, beta=c.beta,
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
                            # ========== Compute the filter ==========
                            Wk[k] = self.filtup(
                                tRyy, tRnn,
                                gevd=c.gevd if not c.gevdJustForDANSE else True,
                                gevdRank=scn.Qdk[k]
                            )

                            if alg.startswith("rsdanse"):
                                # For rS-DANSE, we apply a relaxation
                                if c.scmEstimation == 'online':
                                    alpha = 1 / np.log10(iEff + 10)
                                else:
                                    alpha = 1 / np.log10(i + 10)
                                Wk[k][..., :c.Mk[k], :scn.Qdk[k]] = (1 - alpha) * WkkPrev_rS[k] +\
                                    alpha * Wk[k][..., :c.Mk[k], :scn.Qdk[k]]
                                WkkPrev_rS[k] = Wk[k][..., :c.Mk[k], :scn.Qdk[k]]

                                if k == 0:
                                    iEff += 1  # Increment effective iteration index
                            
                            elif c.scmEstimation == 'online':
                                iEff += 1  # effective iteration index for online-mode
                        else:
                            # No update for this node
                            if alg.startswith("tidanse"): # and c.scmEstimation == 'online':
                                # For TI-DANSE, apply the normalization factor
                                Wk[k] = np.linalg.inv(Nk[k]) @ Wk[k]  # loaded from previous iteration/frame
                        
                        # Compute the fusion matrix Pk
                        if alg.startswith("tidanse"):
                            Pk[k] = Wk[k][..., :c.Mk[k], :c.Qd] @\
                                np.linalg.inv(Wk[k][..., c.Mk[k]:, :c.Qd])
                        else:
                            Pk[k] = Wk[k][..., :c.Mk[k], :scn.Qdk[k]]

                        # Store the network-wide filter for this iteration/frame
                        W_netWide[alg][k].append(Ck @ Wk[k][..., :c.D])
                        
                    # Update the normalization factor for TI-DANSE
                    if alg.startswith("tidanse"): # and c.scmEstimation == 'online':
                        # Update anyway, always, at the reference node
                        r = c.refNodeForTInorm
                        tWr = self.filtup(tRyyPrev[r], tRnnPrev[r], gevd=c.gevd, gevdRank=c.Qd)
                        gamma = tWr[..., c.Mk[k]:, :c.Qd]

                    # Update the updating node index for next iteration
                    if onlineModeCriterion:
                        u = (u + 1) % c.K
                
                # Store the iterative variables for the next frame
                ivOut[alg] = {
                    'Pk': Pk,
                    'WkkPrev_rS': WkkPrev_rS,
                    'Wk': Wk,  # Last filter for each node
                    'u': u,
                    'i': iEff,  # effective iteration index
                    'tRyy': tRyyPrev,
                    'tRss': tRssPrev,
                    'tRnn': tRnnPrev,
                    'nUpdates_tRyy': nUpdates_tRyy,
                    'nUpdates_tRnn': nUpdates_tRnn,
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


        if gevd:
            sigFull = np.zeros((Ryy.shape[0], Ryy.shape[1]), dtype=complex)
            Xfull = np.zeros((Ryy.shape[0], Ryy.shape[1], Ryy.shape[1]), dtype=complex)
            for f in range(Ryy.shape[0]):
                try:
                    sigma, Xmat = sla.eigh(Ryy[f, ...], Rnn[f, ...])
                except np.linalg.LinAlgError:
                    print(f"\nGEVD failed at bin #{f}, aborting GEVD entirely and computing regular MWF instead.", end='\r')
                    return regular_mwf(Ryy, Rss, c.mu)[..., :finalSize]

                idx = np.flip(np.argsort(sigma))
                sigFull[f, :] = sigma[idx]
                Xfull[f, ...] = Xmat[:, idx]
            # Inverse-hermitian of `Xmat`
            Qmat = np.linalg.inv(Xfull.conj().transpose(0, 2, 1))
            # GEVLs tensor - low-rank approximation is done here
            Dmat = np.zeros_like(Ryy, dtype=complex)
            for r in range(np.amin((gevdRank, sigFull.shape[1]))):
                Dmat[:, r, r] = np.squeeze(1 - 1 / sigFull[:, r])
            # Compute filters
            w = Xfull @ Dmat @ Qmat.conj().transpose(0, 2, 1)
            return w[..., :finalSize]
        else:
            tmp = regular_mwf(Ryy, Rss, c.mu)

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


def regular_mwf(Ryy, Rss, mu):
    try:
        tmp = np.linalg.solve(mu * Ryy + (1 - mu) * Rss, Rss)
        # tmp = np.linalg.inv(mu * Ryy + (1 - mu) * Rss) @ Rss
    except np.linalg.LinAlgError:
        print("\nMatrix inversion failed, using pseudo-inverse instead.", end='\r')
        tmp = np.linalg.pinv(mu * Ryy + (1 - mu) * Rss) @ Rss
    return tmp


def matmul_CRC(C: np.ndarray, R: np.ndarray):
    # (F, M, M) @ (F, M, N) → (F, M, N)
    tmp = np.matmul(R, C)
    # (F, N, M) @ (F, M, N) → (F, N, N)
    return np.matmul(C.conj().transpose(0, 2, 1), tmp)
