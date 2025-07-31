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
import numpy as np
from pathlib import Path
from tools.algos import herm
import matplotlib.pyplot as plt
from mypystoi import stoi_any_fs
from tools.base import Parameters
from screeninfo import get_monitors
from dataclasses import dataclass, field

import matplotlib
# matplotlib.use('TkAgg')  # or 'Qt5Agg' (must be set before importing pyplot)

baseResultsDir = f'{Path(__file__).parent}/out'  # Base directory for results

# resDir = f'{baseResultsDir}/res_20250731_1241_online_dynamic_noise_wola'  # specific directory
resDir = 'latest'  # <-- pick the latest results directory

EXPORT = False  # If True, export the figures to files
FORCE_RECOMPUTE_METRICS = True  # If True, recompute metrics even if they exist
# FORCE_RECOMPUTE_METRICS = False  # If True, recompute metrics even if they exist
# METRICS_OVER_FIRST_SECONDS = None  # Number of seconds to consider for waveform-based metrics computation
METRICS_OVER_FIRST_SECONDS = 2  # Number of seconds to consider for waveform-based metrics computation

# WHICH_NODES = 'all'  # 'all' or a list of node indices to process
WHICH_NODES = [0]  # 'all' or a list of node indices to process

# COMPUTE_METRICS_EVERY_N_FRAMES = 10  # Compute metrics every N frames (for online processing)
COMPUTE_METRICS_EVERY_N_FRAMES = 30  # Compute metrics every N frames (for online processing)

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

BYPASS_STOI = True  # If True, bypass STOI computation (useful for debugging)

n_per_col = 2  # Number of figures per column
margin = 100  # Margin between figures in pixels

def main(resDir=resDir, metricsOver=METRICS_OVER_FIRST_SECONDS, bypassStoi=BYPASS_STOI):
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

    # Derive appropriate figure size and position based on screen resolution
    screen_index = get_largest_screen_index()
    monitors = get_monitors()
    if screen_index >= len(monitors):
        raise ValueError(f"Invalid screen_index {screen_index}. Only {len(monitors)} screens detected.")
    screen = monitors[screen_index]
    screen_x, screen_y = screen.x, screen.y
    screen_w, screen_h = screen.width, screen.height
    num_cols = n_per_col
    num_rows = np.ceil(len(listOfFiles) / num_cols)
    # num_cols = np.ceil(len(listOfFiles) / num_rows)
    fig_w = (screen_w - (num_cols + 1) * margin) // num_cols
    fig_h = np.amin([(screen_h - (num_rows + 1) * margin) // num_rows, fig_w / 2])

    for i, file in enumerate(listOfFiles):

        print(f"Loading results from {file}...")
        with open(file, 'rb') as f:
            results = pickle.load(f)

        # Process the results
        c: Parameters = results['cfg']  # Configuration parameters
        
        # Compute metrics over the first seconds of the signal
        if c.scmEstimation != 'online':
            metricsMethod = 'entire_signal'  # Use entire signal for batch processing
        elif c.dynamics != 'static':
            metricsMethod = 'recent_seconds'  # Use recent seconds for non-static dynamics
        else:
            metricsMethod = METRICS_METHOD  # Use the specified method for online processing
        
        if metricsMethod == 'recent_seconds':
            bypassStoi = True  # Bypass STOI computation for 'recent_seconds' method

        pp = PostProcessor(cfg=c)
        
        # Check if metrics have already been computed
        metricsFileName = file.stem + '_metrics.pkl'
        metricsFile = resDir / metricsFileName
        if not FORCE_RECOMPUTE_METRICS and metricsFile.exists():
            print(f"Metrics already computed for {file.stem}, loading from {metricsFileName}...")
            with open(metricsFile, 'rb') as f:
                metrics = pickle.load(f)
        else:
            # Metrics to compute
            metricsToCompute = ['msew', 'msed', 'snr', 'ser']
            if not bypassStoi and c.domain == 'wola' and\
                c.singleLine is None and c.desSigType == 'speech':  # speech enhancement scenario
                metricsToCompute += ['stoi']
            if c.scmEstimation == 'online':
                metricsToCompute.remove('msew')  # msew is not computed in online mode
            
            s = results['s']  # desired signals
            n = results['n']  # noise signals
            y = s + n  # observed signals
            d = np.array([s[c.Mk * k:c.Mk * k + c.D, ...] for k in range(c.K)])  # target signals

            print("\nComputing metrics...")
            t0 = time.time()
            metrics = pp.get_metrics(
                results['W_netWide'],
                y, d, n, s,
                metricsToCompute,
                metricsOver,
                metricsMethod
            )
            print(f"\nMetrics computed in {time.time() - t0:.2f} seconds.")

            # Export metrics to file
            with open(metricsFile, 'wb') as f:
                pickle.dump(metrics, f)

        # Derive figure positioning parameters
        row = i // num_cols
        col = i % num_cols
        pos_x = screen_x + margin + col * (fig_w + 10)
        pos_y = screen_y + margin + row * (fig_h + 10)
        # Post-process results
        fig = pp.plot_metrics(metrics, placement=[pos_x, pos_y, fig_w, fig_h])
        if EXPORT:
            fig.savefig(f"{c.outputDir}/metrics_{file.stem}.svg", dpi=300)
            fig.savefig(f"{c.outputDir}/metrics_{file.stem}.png", dpi=300)

    plt.show(block=False)  # Show all figures
    print("Post-processing completed.")
    return 0

@dataclass
class PostProcessor:
    cfg: Parameters = field(default_factory=lambda: Parameters())

    def get_metrics(
            self,
            W_netWide,
            y, d, n=None, s=None,
            metricsToCompute=[],
            metricsOver=None,
            metricsMethod=None
        ):
        c = self.cfg

        def _apply_filter(Wk, x):
            if c.domain == 'wola':
                return c.get_istft((herm(Wk) @ x.transpose(1, 0, 2)).transpose(1, 0, 2))
            elif 'time' in c.domain:
                return Wk.T.conj() @ x

        def _process(Wk, hWk=None, dk=None, dhatk=None, shatk=None, nhatk=None):
            metrics_curr = dict([(metric, None) for metric in metricsToCompute])
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
                if m == 'stoi':
                    metrics_curr['stoi'] = stoi_any_fs(dk, dhatk, fs_sig=c.fs)
            return metrics_curr
        
        def _processing_loop(k, WcurrFrame, dkTD, silent=False):
            hWk = WcurrFrame['centralized'][k]
            basekwargs = dict(hWk=hWk, dk=dkTD[k])  # Common arguments for metric computation
            kwargs = dict([(alg, basekwargs.copy()) for alg in c.algos])

            metricsCurrAlg = dict([(alg, []) for alg in c.algos])
            for alg in c.algos:
                if not silent:
                    print(f"Computing metrics for {alg}, node {k+1}/{len(nodesToProcess)}...", end='\r')

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
                    metricsCurrAlg[alg].append(_process(wCurr, **kwargs[alg]))

                    if 0 and ii == len(WcurrFrame[alg][k]) - 1:
                        fig, axes = plt.subplots(1, 3)
                        fig.set_size_inches(8.5, 3.5)
                        # Ensure all axes have the same color scale
                        vmin = np.amin([
                            np.amin(20 * np.log10(np.abs((herm(wCurr) @ yc.transpose(1, 0, 2)).transpose(1, 0, 2)[0, ...]))),
                            np.amin(20 * np.log10(np.abs(yc[k * c.Mk, ...]))),
                            np.amin(20 * np.log10(np.abs(sc[k * c.Mk, ...])))
                        ])
                        vmax = np.amax([
                            np.amax(20 * np.log10(np.abs((herm(wCurr) @ yc.transpose(1, 0, 2)).transpose(1, 0, 2)[0, ...]))),
                            np.amax(20 * np.log10(np.abs(yc[k * c.Mk, ...]))),
                            np.amax(20 * np.log10(np.abs(sc[k * c.Mk, ...])))
                        ])
                        axes[0].imshow(20*np.log10(np.abs((herm(wCurr) @ yc.transpose(1, 0, 2)).transpose(1, 0, 2)[0, ...])), vmin=vmin, vmax=vmax)
                        axes[0].set_xlabel('Filtered')
                        axes[1].imshow(20*np.log10(np.abs(yc[k * c.Mk, ...])), vmin=vmin, vmax=vmax)
                        axes[1].set_xlabel('Unprocessed')
                        axes[2].imshow(20*np.log10(np.abs(sc[k * c.Mk, ...])), vmin=vmin, vmax=vmax)
                        axes[2].set_xlabel('Target')
                        fig.suptitle(f"STFTs for {alg}, node {k}, frame {ii + 1}/{len(WcurrFrame[alg][k])}")
                        fig.tight_layout()
                        plt.show(block=False)
                    
            if 0:
                nCols = int(np.sqrt(len(c.algos)))
                nRows = int(np.ceil(len(c.algos) / nCols))
                fig, axes = plt.subplots(nRows, nCols, sharex=True, sharey=True)
                fig.set_size_inches(8.5, 3.5)
                for ax, alg in zip(axes.flatten(), c.algos):
                    ax.plot(kwargs[alg]['dhatk'].T)
                    ax.set_title(alg)
                fig.tight_layout()
                plt.show()

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

        if metricsMethod == 'entire_signal':
            # Get metrics signals to be used for all frames
            yc, sc, nc, dkTD = _get_metrics_signals(endTime=metricsOver)

        # Initialize `metrics` dictionary
        nodesToProcess = range(c.K) if WHICH_NODES == 'all' else WHICH_NODES
        nFramesActual = int(np.ceil(len(W_netWide) / COMPUTE_METRICS_EVERY_N_FRAMES))
        if c.scmEstimation == 'online':
            metrics = dict([(metric, dict([
                (alg, [
                    np.zeros(nFramesActual)
                    for _ in nodesToProcess
                ]) for alg in c.algos
            ])) for metric in metricsToCompute])
        else:
            metrics = dict([(metric, dict([
                (alg, [None for _ in nodesToProcess]) for alg in c.algos
            ])) for metric in metricsToCompute])

        # Process data for each node and each algorithm (and each time frame if online mode)
        outFrameIdx = [0 for _ in nodesToProcess]  # Output frame index for each node
        for k in nodesToProcess:
            if isinstance(W_netWide, list) and c.scmEstimation == 'online':
                # Online-mode processing
                for l, w in enumerate(W_netWide):
                    if l % COMPUTE_METRICS_EVERY_N_FRAMES != 0 or l == 0:
                        continue
                    print(f"Computing metrics at node {k+1}/{len(nodesToProcess)}, frame {l + 1}/{len(W_netWide)}...", end='\r')
                    if metricsMethod == 'recent_seconds':
                        if c.domain == 'wola':
                            # Get metrics signals for the current frame
                            yc, sc, nc, dkTD = _get_metrics_signals(
                                startTime=np.amax((0, l * (c.nfft - c.nhop) / c.fs - metricsOver)),
                                endTime=(l + 1) * (c.nfft - c.nhop) / c.fs
                            )
                        elif 'time' in c.domain:
                            # Get metrics signals for the current frame
                            yc, sc, nc, dkTD = _get_metrics_signals(
                                startTime=np.amax((0, l * c.frameDuration / c.fs - metricsOver)),
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
    
    def plot_metrics(self, metrics: dict, placement: list):
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

        if c.domain == 'wola':
            frameLength = c.nfft - c.nhop
        elif 'time' in c.domain:
            frameLength = int(c.frameDuration * c.fs)

        fig, axes = plt.subplots(1, len(metrics.keys()), sharex=True)
        # Convert size from pixels to inches
        for ii, m in enumerate(metrics.keys()):
            ax = axes[ii] if len(metrics.keys()) > 1 else axes
            if m == 'stoi':
                ax.set_ylim(0, 1)
            if any('danse' in alg for alg in c.algos) or c.scmEstimation == 'online':
                # Line plot when including iterative algorithms
                if m in ['msew', 'msed']:
                    ax.set_yscale('log')
                for jj, alg in enumerate(metrics[m].keys()):
                    if m == 'msew' and alg in ['centralized', 'local','unprocessed']:
                        continue
                    if len(metrics[m][alg][0]) > 1:
                        if metrics[m][alg] is not None:
                            if WHICH_NODES == 'all':
                                data = np.mean(metrics[m][alg], axis=0)
                            else:
                                data = np.mean([
                                    m for i, m in enumerate(metrics[m][alg]) if i in WHICH_NODES
                                ], axis=0)
                            ax.plot(data, label=alg, color=colors[alg],
                                    marker=markers[jj % len(markers)],
                                    markerfacecolor='none', markevery=0.1)
                        else:
                            print(f"No {m} data for {alg}, skipping.")
                    else:
                        # Non-iterative algorithms in batch-mode: horizontal lines
                        plot_h(
                            ax,
                            np.mean(metrics[m][alg]) if WHICH_NODES == 'all' else np.mean([
                                m for i, m in enumerate(metrics[m][alg]) if i in WHICH_NODES
                            ], axis=0),
                            label=alg,
                            color=colors[alg],
                            marker=markers[jj % len(markers)],
                        )
                maxX = len(metrics[list(metrics.keys())[0]]['unprocessed'][0]) if c.scmEstimation == 'online' else c.maxDANSEiter
                # Format x-axis
                ax.set_xlim(0, maxX)
                if c.scmEstimation == 'online':
                    ticksInterval = maxX / 5
                    xTicks = np.arange(0, maxX, ticksInterval)
                    ax.set_xticks(xTicks)
                    ax.set_xticklabels(np.round(xTicks * c.frameDuration * COMPUTE_METRICS_EVERY_N_FRAMES, 2))
                    ax.set_xlabel('Time [s]')
                else:
                    ax.set_xlabel('Iteration')
            else:
                # Bar plot when in batch-mode and not including iterative algorithms
                for jj, alg in enumerate(metrics[m].keys()):
                    if m == 'msew' and alg in ['centralized', 'local','unprocessed']:
                        continue
                    ax.bar(jj, np.mean(metrics[m][alg]), label=alg, color=colors[alg])
            if m == 'snr' and ax.get_ylim()[0] < 0:
                # Ensure SNR = 0 dB is visible as a horizontal line
                ax.axhline(y=0, color='0.5')
            # Add legend
            if m == 'stoi':
                ax.legend(loc='lower center')
            if m == 'msed':
                ax.legend(loc='upper center')
            if c.dynamics == 'moving' and c.scmEstimation == 'online':
                # Plot a vertical line every time the scenario changes
                nChanges = int(c.T / c.movingEvery)
                for i in range(nChanges):
                    ax.axvline(x=(i * c.movingEvery * c.fs / frameLength) / COMPUTE_METRICS_EVERY_N_FRAMES, color='0.5', linestyle='--')
            ax.set_title(m.upper())
            if m in ['snr', 'ser']:
                ax.set_ylim(np.nanargmin([
                    np.amax((-10, ax.get_ylim()[0])),
                    np.nanargmin(metrics[m]['unprocessed'])
                ]), None)  # Ensure y-axis starts at 0
        supti = f'{c.observability.upper()}, {c.scmEstimation} SCMs, node(s): {WHICH_NODES}'
        if c.scmEstimation == 'online' and 'betaString' in c.__dict__.keys():
            supti += f', {c.betaString}'
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
        if area > largest_area:
            largest_area = area
            largest_index = i

    return largest_index


if __name__ == '__main__':
    sys.exit(main())