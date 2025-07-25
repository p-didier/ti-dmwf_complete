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
import time
import copy
import numpy as np
from pathlib import Path
from tools.base import *
from tools.algos import *
from pp import main as main_pp

import matplotlib as mpl
mpl.use('TkAgg')  # use TkAgg backend to avoid issues when plotting
import matplotlib.pyplot as plt
plt.ion()  # interactive mode on

PATH_TO_CFG = ".\\config\\cfg.yml"  # Path to the configuration file

TEST_SET = [
    {
        'scmEstimation': 'oracle',
        'observability': 'foss',
    },
    {
        'scmEstimation': 'oracle',
        'observability': 'poss',
    },
    {
        'scmEstimation': 'batch',
        'observability': 'foss',
    },
    {
        'scmEstimation': 'batch',
        'observability': 'poss',
    },
    # {
    #     'scmEstimation': 'online',
    #     'observability': 'foss',
    # },
    # {
    #     'scmEstimation': 'online',
    #     'observability': 'poss',
    # },
]

def main():
    """Main function (called by default when running script)."""
    cfgBase = Parameters()
    cfgBase.load_from_yaml(PATH_TO_CFG)
    cfgBase.outputDir = f"{Path(__file__).parent}\\out\\res_{time.strftime('%Y%m%d_%H%M')}_{cfgBase.suffix}"  # Set output directory
    # Generate folder if it does not exist
    clean_output_dir(cfgBase.outputDir)  # Clean the output directory if it exists

    np.random.seed(cfgBase.seed)
    rngState = np.random.get_state()

    for i, test in enumerate(TEST_SET):
        print(f"\nTest {i + 1}/{len(TEST_SET)}: {test}")
        cfg = copy.deepcopy(cfgBase)
        cfg.outputFilePath = f"{cfgBase.outputDir}\\res_cfg{i + 1}.pkl"  # Unique output file for each test
        for key, value in test.items():
            setattr(cfg, key, value)
        cfg.__post_init__()

        # Reset random state for each test
        np.random.set_state(rngState)

        # Launch the simulation
        Run(cfg).go()

    # Post-processing
    main_pp()

    pass


def clean_output_dir(output_dir):
    # Go through subdirs in the parent of output_dir and eliminate empty subdirs
    parent_dir = Path(output_dir).parent
    for subdir in parent_dir.iterdir():
        if subdir.is_dir() and not any(subdir.iterdir()):
            print(f"Removing empty directory: {subdir.name}")
            subdir.rmdir()
    # Create the output directory if it does not exist
    if not Path(output_dir).exists():
        print(f"Creating output directory: {Path(output_dir).name}")
        Path(output_dir).mkdir(parents=True, exist_ok=True) 


if __name__ == '__main__':
    sys.exit(main())