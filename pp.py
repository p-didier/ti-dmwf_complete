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
from dataclasses import dataclass, field

baseResultsDir = f'{Path(__file__).parent}/out'  # Base directory for results

resDir = f'{baseResultsDir}/res_20250711_1624_test_new_formulation'  # Path to the results directory
# resDir = f'{baseResultsDir}/res_20250704_1631_rerun_td'  # Path to the results directory

EXPORT = False  # If True, export the figures to files

def main():
    """Main function (called by default when running script)."""
    # Load the results from the directory
    listOfFiles = list(Path(resDir).glob('*.pkl'))
    if not listOfFiles:
        print(f"No results found in {resDir}. Please run main.py first.")
        sys.exit(1)
    
    for file in listOfFiles:
        print(f"Loading results from {file}...")
        with open(file, 'rb') as f:
            results = pickle.load(f)

        # Process the results
        c: Parameters = results['cfg']  # Configuration parameters
        # Metrics to compute
        if c.singleLine is not None:
            print(f"Processing only frequency line {c.singleLine} in WOLA domain.")
            metricsToCompute = ['msew', 'msed', 'snr', 'ser']
        else:
            metricsToCompute = ['msew', 'snr', 'stoi', 'ser']
        
        pp = PostProcessor(cfg=c)
        t0 = time.time()
        s = results['s']  # desired signals
        n = results['n']  # noise signals
        y = s + n  # observed signals
        d = np.array([s[c.Mk * k:c.Mk * k + c.D, ...] for k in range(c.K)])  # target signals

        print("\nComputing metrics...")
        metrics = pp.get_metrics(results['W_netWide'], y, d, n, s, metricsToCompute)
        print(f"\nMetrics computed in {time.time() - t0:.2f} seconds.")

        # Post-process results
        fig = pp.plot_metrics(metrics)
        if EXPORT:
            fig.savefig(f"{c.outputDir}/metrics_{file.stem}.svg", dpi=300)
            fig.savefig(f"{c.outputDir}/metrics_{file.stem}.png", dpi=300)
    
    return 0

@dataclass
class PostProcessor:
    cfg: Parameters = field(default_factory=lambda: Parameters())

    def get_metrics(self, W_netWide, y, d, n=None, s=None, metricsToCompute=[]):
        c = self.cfg

        metrics = dict([(metric, dict([
            (alg, [None for _ in range(c.K)]) for alg in c.algos
        ])) for metric in metricsToCompute])

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
        
        if c.domain == 'wola':
            msedOverFrames = 50 # Number of frames to average MSEd over
            # msedOverFrames = np.shape(y)[-1] # Number of frames to average MSEd over
            yc = y[..., :msedOverFrames]  # Centralized signal for MSEd computation
            if 'snr' in metricsToCompute:
                sc = s[..., :msedOverFrames]  # Centralized signal for MSEd computation
                nc = n[..., :msedOverFrames]  # Centralized signal for MSEd computation
            dkTD = [
                c.get_istft(d[k, ..., :msedOverFrames])
                for k in range(c.K)
            ]  # Desired signal for node k
        elif 'time' in c.domain:
            yc, sc, nc, dkTD = y, s, n, [d[k, ...] for k in range(c.K)]

        for k in range(c.K):

            hWk = W_netWide['centralized'][k]
            kwargs = dict(hWk=hWk, dk=dkTD[k])  # Common arguments for metric computation

            for alg in c.algos:
                print(f"Computing metrics for {alg}, node {k}...", end='\r')

                if not isinstance(W_netWide[alg][k], list):
                    W_netWide[alg][k] = [W_netWide[alg][k]]
                
                for ii in range(len(W_netWide[alg][k])):
                    # Compute signal estimates
                    wCurr = W_netWide[alg][k][ii]
                    kwargs['dhatk'] = _apply_filter(wCurr, yc)
                    if 'snr' in metricsToCompute:
                        kwargs['shatk'] = _apply_filter(wCurr, sc)
                        kwargs['nhatk'] = _apply_filter(wCurr, nc)

                    if k == 0 and ii == 5 and c.observability == 'poss' and c.scmEstimation == 'batch':
                        pass

                    # Compute metrics for the current filter
                    metric_curr = _process(wCurr, **kwargs)

                    for m in metricsToCompute:
                        if metrics[m][alg][k] is None:
                            metrics[m][alg][k] = []
                        metrics[m][alg][k].append(metric_curr[m])

        return metrics
    
    def plot_metrics(self, metrics: dict):
        c = self.cfg

        def plot_h(ax, val, label, color='C0', marker='o'):
            """Plot horizontal line."""
            ax.plot(
                np.linspace(0, c.maxDANSEiter - 1, num=10),
                np.full(10, val),
                marker, linestyle='--', color=color,
                markerfacecolor='none',
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

        fig, axes = plt.subplots(1, len(metrics.keys()), sharex=True)
        fig.set_size_inches(8.5, 3.5)
        for ii, m in enumerate(metrics.keys()):
            ax = axes[ii] if len(metrics.keys()) > 1 else axes
            if m == 'stoi':
                ax.set_ylim(0, 1)
            if any('danse' in alg for alg in c.algos):
                # Line plot when including iterative algorithms
                if m in ['msew', 'msed']:
                    ax.set_yscale('log')
                for jj, alg in enumerate(metrics[m].keys()):
                    if m == 'msew' and alg in ['centralized', 'local','unprocessed']:
                        continue
                    if 'danse' in alg:
                        if metrics[m][alg] is not None:
                            if c.observability == 'poss' and c.scmEstimation == 'batch':
                                pass
                            data = np.mean(metrics[m][alg], axis=0)
                            ax.plot(data, label=alg, color=colors[alg],
                                    marker=markers[jj % len(markers)],
                                    markerfacecolor='none')
                        else:
                            print(f"No {m} data for {alg}, skipping.")
                    else:
                        # Non-iterative algorithms as horizontal lines
                        plot_h(
                            ax,
                            np.mean(metrics[m][alg]),
                            label=alg,
                            color=colors[alg],
                            marker=markers[jj % len(markers)]
                        )
                ax.set_xlim(0, c.maxDANSEiter - 1)
            else:
                # Bar plot when not including iterative algorithms
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
            ax.set_title(m)
        fig.suptitle(f'{c.observability}, {c.scmEstimation}')
        fig.tight_layout()
        plt.show(block=False)

        return fig



if __name__ == '__main__':
    sys.exit(main())