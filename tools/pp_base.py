
import yaml
import numpy as np
from pathlib import Path
from typing import Union
from screeninfo import get_monitors
from dataclasses import dataclass, field

@dataclass
class PostProcParameters:
    """A dataclass for simulation parameters."""

    # ==== File and Directory Parameters ====
    baseResultsDir: str = field(default_factory=lambda: f'./out')  # Base directory for results
    resultsDir: str = field(default_factory=lambda: 'latest')  # Results subdirectory

    # ==== Export and Recompute Flags ====
    export: bool = False  # If True, export the figures to files
    forceRecomputeMetrics: bool = True  # If True, recompute metrics even if they exist

    # ==== Metrics Computation Parameters ====
    metricsChunkDuration: float = 2  # Duration of the chunk to compute metrics over (in seconds)
    metricsChunkShift: float = 0.5  # Shift of the chunk to compute metrics over (in seconds)
    cumulatedAverage: bool = False  # If True, apply a cumulative average (smoothing) to the metrics
    metricsOverFirstSeconds: Union[None, float] = 2  # Number of seconds to consider for waveform-based metrics computation
    computeMetricsEveryNFrames: int = 10  # Compute metrics every N frames (for online processing)
    deltasSnrSer: bool = False  # If True, show SNR and SER as deltas from the local estimate
    metricsMethod: str = 'entire_signal'  # Method for metrics computation
    bypassStoi: bool = False  # If True, bypass STOI computation (useful for debugging)
    stoiInterval: Union[list, float] = 0.75  # STOI interval specification
    extendedStoi: bool = False  # If True, use extended STOI computation
    metricsToComputeOverride: Union[None, list] = None  # If not None, override the metrics to compute

    # ==== Node Selection Parameters ====
    whichNodes: Union[str, list] = 'all'  # 'all' or a list of node indices to process

    # ==== Plotting Parameters ====
    forcedYlim: dict = field(default_factory=lambda: {
        'msew': [1e-27, 1e6],  # If not None, force y-axis limits for msew
    })
    stairsPlot: bool = False  # If True, plot a metrics stairs plot for each static segment (only used in online mode with dynamic scenarios)
    intelligibilityMetrics: list[str] = field(default_factory=lambda: ['stoi', 'pesq'])
    n_per_col: int = 2  # Number of figures per column after plt.show()
    margin: int = 100  # Margin between figures in pixels

    # ==== Debug Parameters ====
    plotWaveforms: bool = False  # If True, plot waveforms for debugging

    def __post_init__(self):
        if 'stoi' in self.intelligibilityMetrics and self.extendedStoi:
            # Replace 'stoi' with 'estoi'
            self.intelligibilityMetrics = ['estoi' if m == 'stoi' else m for m in self.intelligibilityMetrics]

    def load_from_yaml(self, path: str):
        """Load parameters from a YAML file."""
        with open(path, 'r') as file:
            data = yaml.safe_load(file)
        for key, value in data.items():
            setattr(self, key, value)
        self.__post_init__()

    def get_figsize(self, n):
        """Returns the figure size based on the screen resolution."""
        screen_index = get_largest_screen_index()
        monitors = get_monitors()
        if screen_index >= len(monitors):
            raise ValueError(f"Invalid screen_index {screen_index}. Only {len(monitors)} screens detected.")
        screen = monitors[screen_index]
        screen_x, screen_y = screen.x, screen.y
        screen_w, screen_h = screen.width, screen.height
        num_cols = self.n_per_col
        num_rows = np.ceil(n / num_cols)
        fig_w = (screen_w - (num_cols + 1) * self.margin) // num_cols
        fig_h = np.amin([(screen_h - (num_rows + 1) * self.margin) // num_rows, fig_w / 2])
        return fig_w, fig_h, screen_x, screen_y, num_cols

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
