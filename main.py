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
import itertools
import numpy as np
from pathlib import Path
from tools.base import *
from tools.algos import *
from pp import main as postprocessing


from pesq import pesq

PATH_TO_CFG = f".\\config\\cfg.yml"  # Path to the configuration file
# PATH_TO_CFG = f".\\config\\cfg_tidmwf.yml"  # Path to the configuration file

testParams = {
    # 'scmEstimation': ['oracle'],
    'scmEstimation': ['online'],
    # 'scmEstimation': ['batch'],
    # 'observability': ['foss', 'poss'],
    # 'observability': ['foss'],
    'observability': ['poss'],
    # 'observability': ['gls'],
    # 'truncateRIRsNarrowbandAssumption': [True, False],
    'thresholdObsRelDetectability': [-12, -10, -8],  # [dB]
    # 'thresholdObsRelDetectability': [-12, -8],  # [dB]
    # 'diffuseSNR': [-9999, -5],  # [dB]
    # 'beta': [0.879, 0.938, 0.967],  # [dB]
}

# Build TEST_SET based on testParams
TEST_SET = [
    dict(zip(testParams.keys(), values))
    for values in itertools.product(*testParams.values())
]

def main():
    """Main function (called by default when running script)."""
    cfgBase = Parameters()
    cfgBase.load_from_yaml(PATH_TO_CFG)
    cfgBase.outputDir = f"{Path(__file__).parent}\\out\\res_{time.strftime('%Y%m%d_%H%M')}_{cfgBase.suffix}"  # Set output directory
    # Generate folder if it does not exist
    clean_output_dir(cfgBase.outputDir)  # Clean the output directory if it exists

    tMaster = time.time()
    # print(f"Starting simulation with configuration: {cfgBase}")
    for idxMC in range(cfgBase.nMCruns):
        tMC = time.time()
        if cfgBase.nMCruns > 1:
            trueSeed = cfgBase.seed + idxMC  # Change seed for each Monte Carlo run
        else:
            print("\nSingle run (no Monte Carlo)")
            trueSeed = cfgBase.seed

        np.random.seed(trueSeed)
        rngState = np.random.get_state()

        for i, test in enumerate(TEST_SET):
            cfg = copy.deepcopy(cfgBase)
            setattr(cfg, 'seed', trueSeed)  # Set the seed for the current test
            if cfgBase.nMCruns > 1:
                print(f"\n[MC run {idxMC + 1}/{cfgBase.nMCruns}] Test {i + 1}/{len(TEST_SET)}: {test}")
                cfg.outputFilePath = f"{cfgBase.outputDir}\\res_cfg{i + 1}_mc{idxMC + 1}.pkl"  # Unique output file for each test
            else:
                print(f"\n[Test {i + 1}/{len(TEST_SET)}]: {test}")
                cfg.outputFilePath = f"{cfgBase.outputDir}\\res_cfg{i + 1}.pkl"  # Unique output file for each test
            
            for key, value in test.items():
                setattr(cfg, key, value)
            cfg.__post_init__()

            # Reset random state for each test
            np.random.set_state(rngState)

            # Launch the simulation
            Run(cfg).go()

        if cfgBase.nMCruns > 1:
            print(f"\nMC run {idxMC + 1}/{cfgBase.nMCruns} completed in {time.time() - tMC:.2f} seconds.")
    print(f"\nAll runs completed in {time.time() - tMaster:.2f} seconds.")

    # Post-processing
    postprocessing()

    return 0


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