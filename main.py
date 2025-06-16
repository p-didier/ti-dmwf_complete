# Purpose of script:
# This script is the main entry point for ...
#
# Context:
# ....
#
# Created on: ....
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys
import numpy as np
from tools.algos import *
from tools.base import *

import matplotlib as mpl
mpl.use('TkAgg')  # use TkAgg backend to avoid issues when plotting
import matplotlib.pyplot as plt
plt.ion()  # interactive mode on

PATH_TO_CFG = ".\\config\\cfg.yml"  # Path to the configuration file

def main():
    """Main function (called by default when running script)."""
    cfg = Parameters()
    cfg.load_from_yaml(PATH_TO_CFG)

    np.random.seed(cfg.seed)

    # Launch the simulation
    Run(cfg).launch()
    pass

if __name__ == '__main__':
    sys.exit(main())