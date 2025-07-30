# Purpose of script:
# ...
#
# Context:
# xxx
#
# Created on:
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from screeninfo import get_monitors
import time

def arrange_figures(figures, n_per_col=3, screen_index=0, fig_w=600, fig_h=400):
    """
    Arrange matplotlib figures in a grid on the specified screen.

    Must be called *before* plt.show(), otherwise window positioning may not work.
    """
    monitors = get_monitors()
    if screen_index >= len(monitors):
        raise ValueError(f"Invalid screen_index {screen_index}. Only {len(monitors)} screens detected.")

    screen = monitors[screen_index]
    screen_x, screen_y = screen.x, screen.y

    num_figs = len(figures)
    num_cols = int(np.ceil(num_figs / n_per_col))

    backend = plt.get_backend().lower()

    for idx, fig in enumerate(figures):
        col = idx // n_per_col
        row = idx % n_per_col

        pos_x = screen_x + col * fig_w
        pos_y = screen_y + row * fig_h

        manager = fig.canvas.manager

        # Force GUI initialization
        fig.canvas.draw()
        plt.pause(0.01)

        # TkAgg
        if 'tkagg' in backend:
            try:
                manager.window.wm_geometry(f"+{pos_x}+{pos_y}")
            except Exception as e:
                print(f"Could not move Tk window: {e}")
        # QtAgg, Qt5Agg, etc.
        elif 'qt' in backend:
            try:
                manager.window.setGeometry(pos_x, pos_y, fig_w, fig_h)
            except Exception as e:
                print(f"Could not move Qt window: {e}")
        else:
            print(f"Unsupported backend '{backend}' for window positioning.")

def main():
    """Main function (called by default when running script)."""
    figures = []
    for _ in range(6):
        fig, ax = plt.subplots()
        ax.plot(np.random.rand(10))
        figures.append(fig)

    arrange_figures(figures, n_per_col=2, screen_index=0)

    # Only call plt.show() after arranging
    plt.show()

if __name__ == '__main__':
    sys.exit(main())