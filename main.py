# Purpose of script:
# This script is the main entry point for testing the TI-dMWF algorithm.
#
# Context:
# Writing of (TI-)dMWF paper.
#
# Created on: 16/06/2025
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys
import copy
import numpy as np
from tools.algos import *
from tools.base import *

import matplotlib as mpl
mpl.use('TkAgg')  # use TkAgg backend to avoid issues when plotting
import matplotlib.pyplot as plt
plt.ion()  # interactive mode on

PATH_TO_CFG = ".\\config\\cfg.yml"  # Path to the configuration file

TEST_SET = [
    # {
    #     'scmEstimation': 'oracle',
    #     'observability': 'foss',
    # },
    {
        'scmEstimation': 'oracle',
        'observability': 'poss',
    },
    # {
    #     'scmEstimation': 'batch',
    #     'observability': 'foss',
    # },
    # {
    #     'scmEstimation': 'batch',
    #     'observability': 'poss',
    # },
]

def main():
    """Main function (called by default when running script)."""
    cfgBase = Parameters()
    cfgBase.load_from_yaml(PATH_TO_CFG)

    np.random.seed(cfgBase.seed)
    rngState = np.random.get_state()

    for i, test in enumerate(TEST_SET):
        print(f"\nTest {i + 1}/{len(TEST_SET)}: {test}")
        cfg = copy.deepcopy(cfgBase)
        for key, value in test.items():
            setattr(cfg, key, value)

        # Reset random state for each test
        np.random.set_state(rngState)

        # Launch the simulation
        Run(cfg).launch()

    pass

if __name__ == '__main__':
    sys.exit(main())