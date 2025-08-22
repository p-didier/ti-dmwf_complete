# Purpose of script:
# This script post-processes the results of main.py.
#
# Context:
# Development of the (TI-)dMWF algorithm.
#
# Created on: 02/07/2025
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys
import time
import pickle
import matplotlib
import numpy as np
from pathlib import Path
from tools.algos import herm
import matplotlib.pyplot as plt
from mypystoi import stoi_any_fs
from tools.base import Parameters
from screeninfo import get_monitors
from dataclasses import dataclass, field


baseResultsDir = f'{Path(__file__).parent}/out'  # Base directory for results

resDir = f'{baseResultsDir}/res_20250820_1021_6MCs_correct_speech_upDANSEeveryFrame'  # specific directory
# resDir = 'latest'  # <-- pick the latest results directory

# EXPORT = True  # If True, export the figures to files
EXPORT = False  # If True, export the figures to files

FORCE_RECOMPUTE_METRICS = True  # If True, recompute metrics even if they exist
# FORCE_RECOMPUTE_METRICS = False  # If True, recompute metrics even if they exist

# ===== Used in PostProcessor.get_metrics_from_full_signal() =====
METRICS_CHUNK_DURATION = 3  # Duration of the chunk to compute metrics over (in seconds)
METRICS_CHUNK_SHIFT = 0.5  # Shift of the chunk to compute metrics over (in seconds)
# CUMULATED_AVERAGE = True  # If True, apply a cumulative average (smoothing) to the metrics
CUMULATED_AVERAGE = False  # If True, apply a cumulative average (smoothing) to the metrics
# ================================================================

# ===== Used in PostProcessor.get_metrics() =====
# METRICS_OVER_FIRST_SECONDS = None  # Number of seconds to consider for waveform-based metrics computation
METRICS_OVER_FIRST_SECONDS = 2  # Number of seconds to consider for waveform-based metrics computation
COMPUTE_METRICS_EVERY_N_FRAMES = 10  # Compute metrics every N frames (for online processing)
# COMPUTE_METRICS_EVERY_N_FRAMES = 30  # Compute metrics every N frames (for online processing)
# ================================================================

# DELTAS_SNR_SER = True  # If True, show SNR and SER as deltas from the local estimate
DELTAS_SNR_SER = False  # If True, show SNR and SER as deltas from the local estimate

WHICH_NODES = 'all'  # 'all' or a list of node indices to process
# WHICH_NODES = [0]  # 'all' or a list of node indices to process

# Metrics computation method for online mode:
# - 'entire_signal' to compute metrics over the first `METRICS_OVER_FIRST_SECONDS`
#   seconds of signal using the filter at the frame considered
#   (STOI is computed).
# - 'recent_seconds' to compute metrics over the most recent `METRICS_OVER_FIRST_SECONDS`
#   seconds of signal using the filter at the frame considered
#   (STOI is not computed).
METRICS_METHOD = 'entire_signal'
# METRICS_METHOD = 'recent_seconds'
# NB: 'recent_seconds' can only be meaningfully used for online processing.
#   For batch processing, we use 'entire_signal' by default.

# Overriding parameters
BYPASS_STOI = False  # If True, bypass STOI computation (useful for debugging)
# BYPASS_STOI = True  # If True, bypass STOI computation (useful for debugging)
STOI_INTERVAL = [5, -1]  # Interval over which to compute STOI (in s, -1 = end of signal)
# EXTENDED_STOI = False  # If True, use extended STOI computation
EXTENDED_STOI = True  # If True, use extended STOI computation
METRICS_TO_COMPUTE_OVERRIDE = None  # If not None, override the metrics to compute
# METRICS_TO_COMPUTE_OVERRIDE = ['msew']
FORCED_YLIM = {
    'msew': [1e-27, 1e6],  # If not None, force y-axis limits for msew
}

n_per_col = 2  # Number of figures per column after plt.show()
margin = 100  # Margin between figures in pixels

def main(resDir=resDir):
    """Main function (called by default when running script)."""
    # Load the results from the directory
    if resDir == 'latest':
        # Find the latest results directory
        listOfDirs = sorted(Path(baseResultsDir).glob('res_*'), key=lambda x: x.stat().st_birthtime, reverse=True)
        if not listOfDirs:
            print("No results directories found.")
            sys.exit(1)
        resDir = listOfDirs[0]  # Take the most recent directory
    else:
        resDir = Path(resDir)
    
    listOfFiles = list(resDir.glob('*.pkl'))
    listOfFiles = [file for file in listOfFiles if not file.stem.endswith('_metrics')]
    if not listOfFiles:
        print(f"No results found in {resDir}. Please run main.py first.")
        sys.exit(1)
    if any([file.stem.endswith('_comb') for file in listOfFiles]):
        # Only use combined files
        listOfFiles = [file for file in listOfFiles if file.stem.endswith('_comb')]

    # Group files by CFG number
    groupedFiles = {}
    for file in listOfFiles:
        cfgRef = '_'.join(file.stem.split('_')[:2])  # Extract the CFG number from the file name
        groupedFiles.setdefault(cfgRef, []).append(file)
    print(f"Grouped files by CFG number.")

    # Derive appropriate figure size and position based on screen resolution
    fig_w, fig_h, screen_x, screen_y, num_cols = get_figsize(n=len(groupedFiles))

    for i, (cfgRef, files) in enumerate(groupedFiles.items()):

        # Check if metrics have already been computed
        metricsFileName = cfgRef + '_metrics.pkl'
        metricsFile = resDir / metricsFileName
        if not FORCE_RECOMPUTE_METRICS and metricsFile.exists():
            print(f"Metrics already computed for {cfgRef}, loading from {metricsFileName}...")
            with open(metricsFile, 'rb') as f:
                metrics = pickle.load(f)
            
            # Set up basis for PostProcessor
            with open(files[0], 'rb') as f:
                results = pickle.load(f)
            c: Parameters = results['cfg']  # Configuration parameters
            pp = PostProcessor(cfg=c)
        else:
            metrics = []
            for idxMC, f in enumerate(files):
                print(f"Loading results from {f.name} ({idxMC + 1}/{len(files)})...")
                with open(f, 'rb') as f:
                    results = pickle.load(f)
                # Process the results
                c: Parameters = results['cfg']  # Configuration parameters
                pp = PostProcessor(cfg=c)
                
                if METRICS_TO_COMPUTE_OVERRIDE is not None:
                    metricsToCompute = METRICS_TO_COMPUTE_OVERRIDE
                else:
                    metricsToCompute = ['msew', 'msed', 'snr', 'ser']
                    if not pp.bypassStoi and c.domain == 'wola' and\
                        c.singleLine is None and c.desSigType == 'speech':  # speech enhancement scenario
                        metricsToCompute += ['stoi']
                    if c.scmEstimation == 'online':
                        metricsToCompute.remove('msed')  # msed is not computed in online mode
                        metricsToCompute.remove('msew')  # msew is not computed in online mode

                if DELTAS_SNR_SER and ('snr' in metricsToCompute or 'ser' in metricsToCompute) and \
                    'local' not in c.algos:
                    raise ValueError("DELTAS_SNR_SER is enabled but 'local' algorithm is not included.")

                if 's' in results.keys():
                    # Format 1: time-domain raw mic signals and filter weights
                    dataIn = {
                        'W_netWide': results['W_netWide'],
                        's': results['s'],
                        'n': results['n'],
                        'y': results['s'] + results['n'],
                        'd': np.array([results['s'][c.Mkc[k]:c.Mkc[k] + c.D, ...] for k in range(c.K)])
                    }
                elif 'shatk' in results.keys():
                    # Format 2: time-domain processed signals
                    dataIn = {
                        'd': results['d'],
                        'shatk': results['shatk'],
                        'nhatk': results['nhatk']
                    }
                    if 1:
                        plot_signals(dataIn, c)
                else:
                    raise ValueError("Unknown results format (no 's' or 'shat' key in results dict).")

                print("\nComputing metrics...")
                t0 = time.time()
                if c.scmEstimation == 'online' and c.domain == 'wola':
                    fcn = pp.get_metrics_from_full_signal
                else:
                    fcn = pp.get_metrics
                metrics.append(fcn(
                    dataIn,
                    metricsToCompute,
                ))
                print(f"\nMetrics for file (MC run) {idxMC + 1}/{len(files)} computed in {time.time() - t0:.2f} seconds.")


            # Rearrange metrics: place MC runs in the last dimension of the deepest level
            metricsRearranged = dict([(m, dict([(alg, []) for alg in c.algos])) for m in metricsToCompute])
            for m in metricsToCompute:
                for alg in c.algos:
                    for k in pp.nodesToProcess:
                        metricsRearranged[m][alg].append(
                            np.array([mm[m][alg][k] for mm in metrics])
                        )
                    # Convert to numpy array
                    metricsRearranged[m][alg] = np.array(metricsRearranged[m][alg])
            metrics = metricsRearranged

            # Export metrics to file
            with open(metricsFile, 'wb') as f:
                pickle.dump(metrics, f)

        if c.observability == 'poss' and 'CFs' in c.__dict__.keys():
            print(c.CFs) 

        # Derive figure positioning parameters
        row = i // num_cols
        col = i % num_cols
        pos_x = screen_x + margin + col * (fig_w + 10)
        pos_y = screen_y + margin + row * (fig_h + 10)
        # Post-process results
        fig = pp.plot_metrics(
            metrics,
            placement=[pos_x, pos_y, fig_w, fig_h],
            nMC=len(files)
        )
        if EXPORT:
            # Prompt user: are you sure?
            confirm = input(f"☝️ Are you sure you want to export metrics figures to {Path(c.outputDir).stem}? (y/n) ")
            if confirm.lower() != 'y':
                print("Export cancelled.")
                continue
            fig.savefig(f"{resDir}/metrics_{cfgRef}.svg", dpi=300)
            fig.savefig(f"{resDir}/metrics_{cfgRef}.png", dpi=300)
            print(f"Metrics figures exported to {resDir}/metrics_{cfgRef}.svg and .png")

    plt.show(block=False)  # Show all figures
    print("Post-processing completed.")
    return 0


def plot_signals(sigs, c: Parameters, kPlot=0):
    """Plot the signals."""
    dhatk = [
        dict([(alg, sigs['shatk'][k][alg] + sigs['nhatk'][k][alg]) for alg in c.algos])
        for k in range(c.K)
    ]
    dk = sigs['d']
    fig, axes = plt.subplots(len(c.algos), 1, sharex=True, sharey=True)
    for alg_idx, alg in enumerate(c.algos):
        axes[alg_idx].plot(dhatk[kPlot][alg][0, :], 'c')
        axes[alg_idx].plot(dk[kPlot, 0, :], 'k', alpha=0.5)
        axes[alg_idx].set_ylabel(alg)
        axes[alg_idx].set_ylim((np.amin(dk[kPlot, 0, :]) - 0.1, np.amax(dk[kPlot, 0, :]) + 0.1))
    fig.tight_layout()
    plt.show(block=False)
    if 0:
        import simpleaudio as sa
        audio_array *= 32767 / max(abs(audio_array))
        audio_array = audio_array.astype(np.int16)
        sa.play_buffer(audio_array,1,2,c.fs)
    pass


@dataclass
class PostProcessor:
    cfg: Parameters = field(default_factory=lambda: Parameters())

    def __post_init__(self):
        c = self.cfg
        if c.scmEstimation != 'online':
            self.metricsMethod = 'entire_signal'  # Use entire signal for batch processing
        elif c.dynamics != 'static':
            self.metricsMethod = 'recent_seconds'  # Use recent seconds for non-static dynamics
        else:
            self.metricsMethod = METRICS_METHOD  # Use the specified method for online processing
        if self.metricsMethod == 'recent_seconds':
            self.bypassStoi = True  # Bypass STOI computation for 'recent_seconds' method
        else:
            self.bypassStoi = BYPASS_STOI
        self.nodesToProcess = range(c.K) if WHICH_NODES == 'all' else WHICH_NODES
        if isinstance(c.Mk, int):
            self.Mk = [c.Mk] * c.K
            self.Mkc = np.cumsum([0] + self.Mk)

    def get_metrics_from_full_signal(
            self,
            dataIn,
            metricsToCompute=[],
        ):
        c = self.cfg

        if 's' in dataIn.keys():
            W_netWide = dataIn['W_netWide']
            s = dataIn['s']
            n = dataIn['n']
            y = dataIn['y']
            d = dataIn['d']
            # Compute estimated signals
            dhatk = dict([(k, dict([(alg, np.zeros(d[k, ...].shape, dtype=y.dtype)) for alg in c.algos])) for k in self.nodesToProcess])
            shatk = dict([(k, dict([(alg, np.zeros(d[k, ...].shape, dtype=y.dtype)) for alg in c.algos])) for k in self.nodesToProcess])
            nhatk = dict([(k, dict([(alg, np.zeros(d[k, ...].shape, dtype=y.dtype)) for alg in c.algos])) for k in self.nodesToProcess])
            for k in self.nodesToProcess:
                if isinstance(W_netWide, list) and c.scmEstimation == 'online':
                    # Online-mode processing
                    for l, w in enumerate(W_netWide):
                        yc = y[:, :, l]
                        sc = s[:, :, l]
                        nc = n[:, :, l]
                        for alg in c.algos:
                            if c.domain == 'wola':
                                Wk = w[alg][k]
                                if isinstance(Wk, list):
                                    # If Wk is a list, we take the last element (the most recent filter)
                                    Wk = Wk[-1]
                                dhatk[k][alg][..., l] = np.einsum('ijk,ki->ij', herm(Wk), yc)
                                shatk[k][alg][..., l] = np.einsum('ijk,ki->ij', herm(Wk), sc)
                                nhatk[k][alg][..., l] = np.einsum('ijk,ki->ij', herm(Wk), nc)
                else:
                    raise NotImplementedError("Offline-mode processing is not implemented.")
        elif 'shatk' in dataIn.keys():
            d = np.array(dataIn['d'])
            shatk = dataIn['shatk']
            nhatk = dataIn['nhatk']
            dhatk = [
                dict([(alg, shatk[k][alg] + nhatk[k][alg]) for alg in c.algos])
                for k in range(c.K)
            ]
        
        wolaFlag = c.domain == 'wola' and c.singleLine is not None
        if wolaFlag:
            # We are in the WOLA domain
            nFramesPerChunkShift = int(np.ceil(METRICS_CHUNK_SHIFT * c.fs / (c.nfft - c.nhop)))
            nFramesPerChunk = int(np.ceil(METRICS_CHUNK_DURATION * c.fs / (c.nfft - c.nhop)))
            nChunks = int(np.ceil(y.shape[-1] / nFramesPerChunkShift))
        else:
            # We are in the time domain
            nSamplesPerChunkShift = int(np.ceil(METRICS_CHUNK_SHIFT * c.fs))
            nSamplesPerChunk = int(np.ceil(METRICS_CHUNK_DURATION * c.fs))
            nChunks = int(np.ceil(list(dhatk[0].values())[0].shape[-1] / nSamplesPerChunkShift))

        if c.scmEstimation == 'online':
            metrics = dict([(metric, dict([
                (alg, [
                    []
                    for _ in self.nodesToProcess
                ]) for alg in c.algos
            ])) for metric in metricsToCompute])
        else:
            metrics = dict([(metric, dict([
                (alg, [None for _ in self.nodesToProcess]) for alg in c.algos
            ])) for metric in metricsToCompute])

        for ii in range(nChunks):
            if wolaFlag:
                idxEnd = ii * nFramesPerChunkShift
                idxBeg = max(idxEnd - nFramesPerChunk, 0)
            else:
                idxEnd = ii * nSamplesPerChunkShift
                idxBeg = max(idxEnd - nSamplesPerChunk, 0)
            for k in self.nodesToProcess:
                dk_chunk = d[k, ..., idxBeg:idxEnd]
                for alg in c.algos:
                    dhatk_chunk = dhatk[k][alg][..., idxBeg:idxEnd]
                    shatk_chunk = shatk[k][alg][..., idxBeg:idxEnd]
                    nhatk_chunk = nhatk[k][alg][..., idxBeg:idxEnd]
                    metrics_curr = self.compute_metrics(
                        metricsToCompute=metricsToCompute,
                        dk=dk_chunk, dhatk=dhatk_chunk,
                        shatk=shatk_chunk, nhatk=nhatk_chunk
                    )
                    for m in metrics_curr.keys():
                        metrics[m][alg][k].append(metrics_curr[m])
        
        # Compute STOI
        if 'stoi' in metricsToCompute:
            idxBeg = int(STOI_INTERVAL[0] * c.fs)
            idxEnd = int(STOI_INTERVAL[1] * c.fs) if STOI_INTERVAL[1] != -1 else -1
            for k in self.nodesToProcess:
                for alg in c.algos:
                    print(f"Computing STOI for {alg}, node {k + 1}/{len(self.nodesToProcess)}...", end='\r')
                    metrics['stoi'][alg][k] = stoi_any_fs(
                        d[k, 0, idxBeg:idxEnd],
                        dhatk[k][alg][0, idxBeg:idxEnd],
                        fs_sig=c.fs,
                        extended=EXTENDED_STOI
                    )

        if CUMULATED_AVERAGE:
            # Apply smoothing to the metrics
            for m in metricsToCompute:
                for alg in c.algos:
                    for k in self.nodesToProcess:
                        if 0:
                            # Full cumulated averaging (all data)
                            metrics[m][alg][k] = np.cumsum(
                                metrics[m][alg][k]
                            ) / np.arange(1, len(metrics[m][alg][k]) + 1)
                        else:
                            # Cumulative average over the last `nFramesPerChunk` frames
                            metrics[m][alg][k] = np.convolve(
                                metrics[m][alg][k],
                                np.ones(nFramesPerChunk) / nFramesPerChunk,
                                mode='valid'
                            )

        if 0:
            nCols = int(np.sqrt(len(c.algos)))
            nRows = int(np.ceil(len(c.algos) / nCols))
            fig, axes = plt.subplots(nRows, nCols, sharex=True, sharey=True)
            fig.set_size_inches(8.5, 3.5)
            for i, alg in enumerate(c.algos):
                ax = axes[i // nCols, i % nCols]
                ax.plot(np.abs(dhatk[0][alg][0, 0, :]), 'b', label='dhatk')
                ax.plot(np.abs(d[0, 0, 0, :]), 'k', label='dk')
                ax.legend()
                ax.set_title(f"{alg} - Node {0 + 1}")
            fig.tight_layout()
            plt.show()
        
        return metrics

    def get_metrics(
            self,
            dataIn,
            metricsToCompute=[],
        ):
        c = self.cfg

        if 's' in dataIn.keys():
            W_netWide = dataIn['W_netWide']
            s = dataIn['s']
            n = dataIn['n']
            y = dataIn['y']
            d = dataIn['d']
        else:
            raise NotImplementedError('Format 2 (processed signals) not implemented.')

        def _apply_filter(Wk, x):
            if c.domain == 'wola':
                return c.get_istft((herm(Wk) @ x.transpose(1, 0, 2)).transpose(1, 0, 2))
            elif 'time' in c.domain:
                return Wk.T.conj() @ x
        
        def _processing_loop(k, WcurrFrame, dkTD, silent=False):
            hWk = WcurrFrame['centralized'][k]
            basekwargs = dict(hWk=hWk, dk=dkTD[k])  # Common arguments for metric computation
            kwargs = dict([(alg, basekwargs.copy()) for alg in c.algos])

            metricsCurrAlg = dict([(alg, []) for alg in c.algos])
            for alg in c.algos:
                if not silent:
                    print(f"Computing metrics for {alg}, node {k+1}/{len(self.nodesToProcess)}...", end='\r')

                if not isinstance(WcurrFrame[alg][k], list):
                    WcurrFrame[alg][k] = [WcurrFrame[alg][k]]
                
                for ii in range(len(WcurrFrame[alg][k])):
                    # Compute signal estimates
                    wCurr = WcurrFrame[alg][k][ii]
                    kwargs[alg]['dhatk'] = _apply_filter(wCurr, yc)
                    if 'snr' in metricsToCompute:
                        kwargs[alg]['shatk'] = _apply_filter(wCurr, sc)
                        kwargs[alg]['nhatk'] = _apply_filter(wCurr, nc)
                    # Compute metrics for the current filter
                    metricsCurrAlg[alg].append(self.compute_metrics(
                        metricsToCompute=metricsToCompute,
                        Wk=wCurr,
                        **kwargs[alg],
                    ))

            return metricsCurrAlg
        
        def _get_metrics_signals(startTime=0, endTime=None):
            """Get metrics signals based on the processing domain."""
            # Get metrics signals to be used for all frames
            if endTime is None:
                idxEnd = -1
            if c.domain == 'wola':
                idxBeg = int(startTime * c.fs / (c.nfft - c.nhop))
                if endTime is not None:
                    idxEnd = int(endTime * c.fs / (c.nfft - c.nhop))
                yc = y[..., idxBeg:idxEnd]  # Centralized signal for MSEd computation
                if 'snr' in metricsToCompute:
                    sc = s[..., idxBeg:idxEnd]  # Centralized signal for MSEd computation
                    nc = n[..., idxBeg:idxEnd]  # Centralized signal for MSEd computation
                dkTD = [
                    c.get_istft(d[k, ..., idxBeg:idxEnd])
                    for k in range(c.K)
                ]  # Desired signal for node k
            elif 'time' in c.domain:
                idxBeg = int(startTime * c.fs)
                if endTime is not None:
                    idxEnd = int(endTime * c.fs)
                yc, sc, nc,  = y[:, idxBeg:idxEnd], s[:, idxBeg:idxEnd], n[:, idxBeg:idxEnd]
                dkTD = [d[k, :, idxBeg:idxEnd] for k in range(c.K)]
            return yc, sc, nc, dkTD

        if self.metricsMethod == 'entire_signal':
            # Get metrics signals to be used for all frames
            yc, sc, nc, dkTD = _get_metrics_signals(endTime=METRICS_OVER_FIRST_SECONDS)

        # Initialize `metrics` dictionary
        self.nodesToProcess = range(c.K) if WHICH_NODES == 'all' else WHICH_NODES
        nFramesActual = int(np.ceil(len(W_netWide) / COMPUTE_METRICS_EVERY_N_FRAMES))
        if c.scmEstimation == 'online':
            metrics = dict([(metric, dict([
                (alg, [
                    np.zeros(nFramesActual)
                    for _ in self.nodesToProcess
                ]) for alg in c.algos
            ])) for metric in metricsToCompute])
        else:
            metrics = dict([(metric, dict([
                (alg, [None for _ in self.nodesToProcess]) for alg in c.algos
            ])) for metric in metricsToCompute])

        # Process data for each node and each algorithm (and each time frame if online mode)
        outFrameIdx = [0 for _ in self.nodesToProcess]  # Output frame index for each node
        for k in self.nodesToProcess:
            if isinstance(W_netWide, list) and c.scmEstimation == 'online':
                # Online-mode processing
                for l, w in enumerate(W_netWide):
                    if l % COMPUTE_METRICS_EVERY_N_FRAMES != 0 or l == 0:
                        continue
                    print(f"Computing metrics at node {k+1}/{len(self.nodesToProcess)}, frame {l + 1}/{len(W_netWide)}...", end='\r')
                    if self.metricsMethod == 'recent_seconds':
                        if c.domain == 'wola':
                            # Get metrics signals for the current frame
                            yc, sc, nc, dkTD = _get_metrics_signals(
                                startTime=np.amax((0, l * (c.nfft - c.nhop) / c.fs - METRICS_OVER_FIRST_SECONDS)),
                                endTime=(l + 1) * (c.nfft - c.nhop) / c.fs
                            )
                        elif 'time' in c.domain:
                            # Get metrics signals for the current frame
                            yc, sc, nc, dkTD = _get_metrics_signals(
                                startTime=np.amax((0, l * c.frameDuration / c.fs - METRICS_OVER_FIRST_SECONDS)),
                                endTime=(l + 1) * c.frameDuration / c.fs
                            )
                    metricsCurrAlg = _processing_loop(k, w, dkTD, silent=True)
                    for alg in c.algos:
                        for m in metricsToCompute:
                            metrics[m][alg][k][outFrameIdx[k]] = metricsCurrAlg[alg][0][m]  # always only one element in `metricsCurrAlg[alg][m]` list in online-mode
                    outFrameIdx[k] += 1
            else:
                metricsCurrAlg = _processing_loop(k, W_netWide, dkTD)
                for alg in c.algos:
                    for m in metricsToCompute:
                        metrics[m][alg][k] = np.array([
                            mm[m] for mm in metricsCurrAlg[alg]
                        ])
        
        return metrics
    
    def compute_metrics(
            self,
            metricsToCompute,
            Wk=None, hWk=None,
            dk=None, dhatk=None,
            shatk=None, nhatk=None
        ):
        metrics_curr = dict([(metric, None) for metric in metricsToCompute if metric != 'stoi'])
        for m in metricsToCompute:
            if m == 'msew':
                metrics_curr['msew'] = np.mean(np.abs(Wk - hWk) ** 2)
            if m == 'msed':
                metrics_curr['msed'] = np.mean(np.abs(dk - dhatk) ** 2)
            if m == 'snr':
                metrics_curr['snr'] = 20 * np.log10(
                    np.mean(np.abs(shatk) ** 2) /
                    np.mean(np.abs(nhatk) ** 2)
                )
            if m == 'ser':
                metrics_curr['ser'] = 20 * np.log10(
                    np.mean(np.abs(dk) ** 2) /
                    np.mean(np.abs(dk - dhatk) ** 2)
                )
            # if m == 'stoi':
            #     metrics_curr['stoi'] = stoi_any_fs(dk, dhatk, fs_sig=self.cfg.fs)
        return metrics_curr

    def plot_metrics(self, metrics: dict, placement: list, nMC: int):
        c = self.cfg

        def plot_h(ax, val, label, color='C0', marker='o'):
            """Plot horizontal line."""
            ax.plot(
                np.linspace(0, c.maxDANSEiter - 1, num=10),
                np.full(10, val),
                marker, linestyle='--', color=color,
                markerfacecolor='none',
                markevery=0.1,
                label=label
            )

        markers = ['o', 's', 'D', '^', 'v', 'x', '+']
        colors = {
            'centralized': 'r',
            'local': '0.5',
            'unprocessed': '0.7',
            'danse': 'b',
            'tidanse': 'y',
            'rsdanse': 'g',
            'dmwf': 'k',
            'tidmwf': 'm',
        }

        fig, axes = plt.subplots(1, len(metrics.keys()))
        # Convert size from pixels to inches
        for ii, m in enumerate(metrics.keys()):
            ax = axes[ii] if len(metrics.keys()) > 1 else axes
            if m == 'stoi':
                ax.set_ylim(0, 1)
            maxX = metrics[list(metrics.keys())[0]][c.algos[0]].shape[-1] if c.scmEstimation == 'online' else c.maxDANSEiter
            if c.dynamics == 'moving' and c.scmEstimation == 'online':
                # Plot a vertical line every time the scenario changes
                nChanges = int(c.T / c.movingEvery)
                for i in range(nChanges):
                    x = i * c.movingEvery / c.T * maxX
                    ax.axvline(x=x, color='0.5', linestyle='--')
            flagDelta = m in ['snr', 'ser'] and DELTAS_SNR_SER
            if (any('danse' in alg for alg in c.algos) or\
                c.scmEstimation == 'online') and m != 'stoi':
                # Line plot when including iterative algorithms
                if m in ['msew', 'msed']:
                    ax.set_yscale('log')
                for jj, alg in enumerate(metrics[m].keys()):
                    if (m == 'msew' and alg in ['centralized', 'local','unprocessed']) or \
                        (flagDelta and alg in ['local', 'unprocessed']):
                        continue
                    col = colors[alg] if alg in colors.keys() else f'C{jj}'
                    if metrics[m][alg].shape[-1] > 1:
                        if WHICH_NODES == 'all':
                            data = np.mean(metrics[m][alg], axis=(0, 1))
                            if flagDelta:
                                data -= np.mean(metrics[m]['local'], axis=(0, 1))
                        else:
                            data = np.mean([
                                m for i, m in enumerate(metrics[m][alg]) if i in WHICH_NODES
                            ], axis=(0, 1))
                            if flagDelta:
                                data -= np.mean([
                                    m for i, m in enumerate(metrics[m]['local']) if i in WHICH_NODES
                                ], axis=(0, 1))
                        ax.plot(data, label=alg, color=col,
                                marker=markers[jj % len(markers)],
                                markerfacecolor='none', markevery=0.1)
                    else:
                        # Non-iterative algorithms in batch-mode: horizontal lines
                        if WHICH_NODES == 'all':
                            data = np.mean(metrics[m][alg])
                            if flagDelta:
                                data -= np.mean(metrics[m]['local'])
                        else:
                            data = np.mean([
                                m for i, m in enumerate(metrics[m][alg]) if i in WHICH_NODES
                            ])
                            if flagDelta:
                                data -= np.mean([
                                    m for i, m in enumerate(metrics[m]['local']) if i in WHICH_NODES
                                ])
                        plot_h(
                            ax,
                            data,
                            label=alg,
                            color=colors[alg],
                            marker=markers[jj % len(markers)],
                        )
                # Format x-axis
                xAxisStart = int(METRICS_CHUNK_DURATION // METRICS_CHUNK_SHIFT)
                ax.set_xlim(xAxisStart, maxX)
                if c.scmEstimation == 'online':
                    ticksInterval = (maxX - xAxisStart) / 5
                    if c.dynamics == 'moving':
                        ticksInterval = np.amin([ticksInterval, c.movingEvery / c.T * maxX])
                    xTicks = np.arange(xAxisStart, maxX, ticksInterval)
                    ax.set_xticks(xTicks)
                    ax.set_xticklabels(np.round(xTicks / (maxX - xAxisStart) * c.T, 2))
                    ax.set_xlabel('Time [s]')
                else:
                    ax.set_xlabel('Iteration')
            else:
                # Bar plot when in batch-mode and not including iterative algorithms
                for jj, alg in enumerate(metrics[m].keys()):
                    col = colors[alg] if alg in colors.keys() else f'C{jj}'
                    if m == 'msew' and alg in ['centralized', 'local','unprocessed']:
                        continue
                    ax.bar(jj, np.mean(metrics[m][alg]), label=alg, color=col)
            if m in ['snr', 'ser'] and ax.get_ylim()[0] < 0:
                # Ensure SNR = 0 dB is visible as a horizontal line
                ax.axhline(y=0, color='0.75')
            # Add legend
            if m == 'stoi':
                ax.legend(loc='lower center')
            if m == 'snr':
                ax.legend(loc='upper center')
            ti = m.upper()
            if m in ['snr', 'ser'] and DELTAS_SNR_SER:
                ti = f'$\\Delta${ti}'
            ax.set_title(ti)
            if m in FORCED_YLIM.keys():
                if FORCED_YLIM[m] is not None:
                    # Force y-axis limits if specified
                    ax.set_ylim(FORCED_YLIM[m])
            # elif m in ['snr', 'ser'] and not CUMULATED_AVERAGE and not DELTAS_SNR_SER:
            #     # Ensure y-axis starts at -10, lowest for SNR and SER
            #     ax.set_ylim(np.nanargmin([
            #         np.amax((-10, ax.get_ylim()[0])),
            #         np.nanargmin(metrics[m][c.algos[0]])
            #     ]), None)
        supti = f'{c.observability.upper()}, {c.scmEstimation} SCMs, node(s): {WHICH_NODES}'
        if c.scmEstimation == 'online' and 'betaString' in c.__dict__.keys():
            supti += f', {c.betaString}'
        if nMC > 1:
            supti += f', #MCs: {nMC}'
        fig.suptitle(supti)

        backend = matplotlib.get_backend().lower()
        manager = fig.canvas.manager
        # Set position
        fig_w, fig_h = placement[2], placement[3]
        if 'tkagg' in backend:
            manager.window.wm_geometry(f"{int(fig_w)}x{int(fig_h)}+{int(placement[0])}+{int(placement[1])}")
        elif 'qt' in backend:
            manager.window.setGeometry(placement[0], placement[1], fig_w, fig_h)
        else:
            print(f"Unsupported backend '{backend}' for window positioning.")


        # === Additional figure: metrics aggregated per static segment as stairs ===
        if c.scmEstimation == 'online' and getattr(c, 'dynamics', None) == 'moving' and hasattr(c, 'movingEvery'):
            # Compute mapping from frame index -> time [s]
            maxX = metrics[list(metrics.keys())[0]]['unprocessed'].shape[-1]
            # Segment edges in time
            seg_edges_time = np.arange(0, c.T + 1e-9, c.movingEvery)
            if seg_edges_time[-1] < c.T:  # ensure we cover the tail
                seg_edges_time = np.append(seg_edges_time, c.T)
            # Convert segment edges to frame indices (the x-domain used for the online plots)
            seg_edges_idx = np.clip(np.round(seg_edges_time / c.T * maxX).astype(int), 0, maxX)
            # Prepare new figure with the same layout/style
            fig2, axes2 = plt.subplots(1, len(metrics.keys()), sharex=True)
            # Position this second figure just below the first one
            placement2 = placement.copy()
            placement2[1] = placement[1] + placement[3] + 15  # below
            backend2 = matplotlib.get_backend().lower()
            manager2 = fig2.canvas.manager
            fig_w2, fig_h2 = placement2[2], placement2[3]
            if 'tkagg' in backend2:
                manager2.window.wm_geometry(f"{int(fig_w2)}x{int(fig_h2)}+{int(placement2[0])}+{int(placement2[1])}")
            elif 'qt' in backend2:
                manager2.window.setGeometry(placement2[0], placement2[1], fig_w2, fig_h2)
            else:
                print(f"Unsupported backend '{backend2}' for window positioning.")
            # Loop over metrics and algorithms to compute per-segment averages and plot as steps
            for ii, m in enumerate(metrics.keys()):
                ax2 = axes2[ii] if len(metrics.keys()) > 1 else axes2
                if m == 'stoi':
                    ax2.set_ylim(0, 1)
                # Vertical lines at segment boundaries
                for s in range(len(seg_edges_idx)):
                    x = seg_edges_idx[s]
                    ax2.axvline(x=x, color='0.5', linestyle='--')
                flagDelta = m in ['snr', 'ser'] and DELTAS_SNR_SER
                # y-values per algorithm: list of length nSegments-1
                for jj, alg in enumerate(metrics[m].keys()):
                    if (m == 'msew' and alg in ['centralized', 'local', 'unprocessed']) or \
                       (flagDelta and alg in ['local', 'unprocessed']):
                        continue
                    # Build the time series data first, averaged over nodes/MC like in the main plot
                    if metrics[m][alg].shape[-1] > 1:
                        if WHICH_NODES == 'all':
                            series = np.mean(metrics[m][alg], axis=(0, 1))
                            if flagDelta:
                                series -= np.mean(metrics[m]['local'], axis=(0, 1))
                        else:
                            series = np.mean([
                                mm for i, mm in enumerate(metrics[m][alg]) if i in WHICH_NODES
                            ], axis=(0, 1))
                            if flagDelta:
                                series -= np.mean([
                                    mm for i, mm in enumerate(metrics[m]['local']) if i in WHICH_NODES
                                ], axis=(0, 1))
                    else:
                        # Non-iterative/batch lines are plotted as constants
                        if WHICH_NODES == 'all':
                            series = np.array([np.mean(metrics[m][alg])])
                            if flagDelta:
                                series -= np.mean(metrics[m]['local'])
                        else:
                            series = np.array([np.mean([
                                mm for i, mm in enumerate(metrics[m][alg]) if i in WHICH_NODES
                            ])])
                            if flagDelta:
                                series -= np.mean([
                                    mm for i, mm in enumerate(metrics[m]['local']) if i in WHICH_NODES
                                ])
                        # Expand constant across all segments
                        series = np.full(len(seg_edges_idx)-1, series.item())
                    # Compute per-segment averages
                    seg_vals = []
                    for s in range(len(seg_edges_idx)-1):
                        beg, end = seg_edges_idx[s], seg_edges_idx[s+1]
                        if end <= beg or beg >= len(series):
                            seg_vals.append(np.nan)
                        else:
                            seg_vals.append(np.nanmean(series[beg:end]))
                    # Build stairs: x as edges, y as segment values
                    x_edges = seg_edges_idx
                    y_vals = np.array(seg_vals)
                    # For log-scale metrics, avoid non-positive values
                    if m in ['msew', 'msed']:
                        y_vals = np.where(y_vals <= 0, np.nan, y_vals)
                        ax2.set_yscale('log')
                    ax2.step(x_edges, np.append(y_vals, y_vals[-1] if len(y_vals) else np.nan),
                             where='post', label=alg, color=colors[alg])
                # Axes formatting mirrors the main plot
                ax2.set_xlim(0, maxX)
                ticksInterval = maxX / 5
                if c.dynamics == 'moving':
                    ticksInterval = np.amin([ticksInterval, c.movingEvery / c.T * maxX])
                xTicks = np.arange(0, maxX + 1e-9, ticksInterval)
                ax2.set_xticks(xTicks)
                ax2.set_xticklabels(np.round(xTicks / maxX * c.T, 2))
                ax2.set_xlabel('Time [s]')
                if m in ['snr', 'ser'] and ax2.get_ylim()[0] < 0:
                    ax2.axhline(y=0, color='0.75')
                ti2 = (f'$\\Delta${m.upper()}' if (m in ['snr', 'ser'] and DELTAS_SNR_SER) else m.upper())
                ax2.set_title(ti2)
                # Legend placement same as main plot
                if m == 'stoi':
                    ax2.legend(loc='lower center')
                if m == 'msed':
                    ax2.legend(loc='upper center')
                if m in FORCED_YLIM.keys() and FORCED_YLIM[m] is not None:
                    ax2.set_ylim(FORCED_YLIM[m])
            supti2 = f'{c.observability.upper()}, {c.scmEstimation} SCMs, node(s): {WHICH_NODES} — static-segment stairs'
            if c.scmEstimation == 'online' and 'betaString' in c.__dict__.keys():
                supti2 += f', {c.betaString}'
            if nMC > 1:
                supti2 += f', #MCs: {nMC}'
            fig2.suptitle(supti2)
            fig2.tight_layout()
        fig.tight_layout()
        return fig


def get_largest_screen_index():
    """Returns the index of the largest screen."""
    monitors = get_monitors()
    # Fallback: return largest screen by area
    largest_index = 0
    largest_area = 0
    for i, monitor in enumerate(monitors):
        area = monitor.width * monitor.height
        if area >= largest_area:
            largest_area = area
            largest_index = i

    return largest_index


def get_figsize(n):
    """Returns the figure size based on the screen resolution."""
    screen_index = get_largest_screen_index()
    monitors = get_monitors()
    if screen_index >= len(monitors):
        raise ValueError(f"Invalid screen_index {screen_index}. Only {len(monitors)} screens detected.")
    screen = monitors[screen_index]
    screen_x, screen_y = screen.x, screen.y
    screen_w, screen_h = screen.width, screen.height
    num_cols = n_per_col
    num_rows = np.ceil(n / num_cols)
    fig_w = (screen_w - (num_cols + 1) * margin) // num_cols
    fig_h = np.amin([(screen_h - (num_rows + 1) * margin) // num_rows, fig_w / 2])
    return fig_w, fig_h, screen_x, screen_y, num_cols


if __name__ == '__main__':
    sys.exit(main())