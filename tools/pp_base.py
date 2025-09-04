
import yaml
import numpy as np
from pathlib import Path
from typing import Union
from screeninfo import get_monitors
from dataclasses import dataclass, field
from collections.abc import Mapping, Sequence

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
    intelligibilityMetrics: list[str] = field(default_factory=lambda: ['stoi', 'pesq'])
    metricsChunkDuration: float = 2  # Duration of the chunk to compute metrics over (in seconds)
    metricsChunkShift: float = 0.5  # Shift of the chunk to compute metrics over (in seconds)
    cumulatedAverage: bool = False  # If True, apply a cumulative average (smoothing) to the metrics
    metricsOverFirstSeconds: Union[None, float] = 2  # Number of seconds to consider for waveform-based metrics computation
    computeMetricsEveryNFrames: int = 10  # Compute metrics every N frames (for online processing)
    deltasSnrSer: bool = False  # If True, show SNR and SER as deltas from the local estimate
    metricsMethod: str = 'entire_signal'  # Method for metrics computation
    bypassStoi: bool = False  # If True, bypass STOI computation (useful for debugging)
    extendedStoi: bool = False  # If True, use extended STOI computation
    metricsToComputeBasis: list[str] = field(default_factory=lambda: ['snr', 'ser', 'msed', 'msew'])
    metricsToComputeOverride: Union[None, list] = None  # If not None, override the metrics to compute
    multiSpeechSourceManagement: str = 'separate'  # 'separate' or 'average' or 'max' for multi-speech source management
    IMchunkType: str = 'single'  # Type of time-chunk to use for intelligibility metric (IM) computation
    #  ^^^ 'single' (one time-chunk, based on `IMinterval`) or 'multi' (multiple chunks, based on `IMmultiChunkDur`)
    IMinterval: Union[list, float] = 0.75  # intelligibility metric (IM) interval specification
    IMmultiChunkDur: float = 1.0  # Duration of each chunk for multi-chunk processing (in seconds)
    IMmultiChunkShiftDur: float = 1.0  # Duration of shift of each chunk for multi-chunk processing (in seconds)

    # ==== Node Selection Parameters ====
    whichNodes: Union[str, list] = 'all'  # 'all' or a list of node indices to process

    # ==== Plotting Parameters ====
    forcedYlim: dict = field(default_factory=lambda: {})
    stairsPlot: bool = False  # If True, plot a metrics stairs plot for each static segment (only used in online mode with dynamic scenarios)
    n_per_col: int = 2  # Number of figures per column after plt.show()
    margin: int = 100  # Margin between figures in pixels

    # ==== Debug Parameters ====
    plotWaveforms: bool = False  # If True, plot waveforms for debugging

    def __post_init__(self):
        if 'stoi' in self.intelligibilityMetrics and self.extendedStoi:
            # Replace 'stoi' with 'estoi'
            self.intelligibilityMetrics = ['estoi' if m == 'stoi' else m for m in self.intelligibilityMetrics]
        if self.multiSpeechSourceManagement == 'max':
            raise NotImplementedError("Max multi-speech source management may not be a valid approach for PODS scenarios.")


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

def _all_same_shape(arrs):
    try:
        s0 = arrs[0].shape
        return all(a.shape == s0 for a in arrs)
    except AttributeError:
        return False

def tree_stack(samples, axis=-1, auto_stack_sequences=True):
    """
    Stack a list of identically-structured trees (dict/list/tuple/arrays) along `axis`.
    Leaves are stacked with np.stack. Intermediate sequences can optionally be stacked
    if they contain arrays of the same shape.
    """
    if not samples:
        raise ValueError("samples must be a non-empty list")

    first = samples[0]

    # Mapping: stack per key
    if isinstance(first, Mapping):
        keys = first.keys()
        return {k: tree_stack([s[k] for s in samples], axis=axis, auto_stack_sequences=auto_stack_sequences)
                for k in keys}

    # Sequence (but not str/bytes): stack per index
    if isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
        # assume equal length/structure
        elems = [tree_stack([s[i] for s in samples], axis=axis, auto_stack_sequences=auto_stack_sequences)
                 for i in range(len(first))]
        # Optionally collapse a list/tuple of arrays into a single array
        if auto_stack_sequences and all(isinstance(e, np.ndarray) for e in elems) and _all_same_shape(elems):
            return np.stack(elems, axis=0)
        # preserve original type
        return type(first)(elems) if isinstance(first, tuple) else elems

    # Leaf: stack values
    return np.stack(samples, axis=axis)
