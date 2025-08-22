# Purpose of script:
# Combine outputs of several simulations before post-processing.
#
# Context:
# dMWF simulations.
#
# Created on: 22/08/2025.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys
import copy
import pickle
import numpy as np
from pathlib import Path

BASE_FOLDER = './out'

COMBINE_FOLDERS = {
    'res_20250820_1021_6MCs_correct_speech_upDANSEeveryFrame': None,
    'res_20250821_1039_DANSEiterevery10_6MCs': {
        'danse': 'danse_10',
        'rsdanse': 'rsdanse_10'
    }
}

def main():
    """Main function (called by default when running script)."""

    if len(COMBINE_FOLDERS) != 2:
        raise ValueError('Currently only supports combining two folders.')
    
    if COMBINE_FOLDERS[list(COMBINE_FOLDERS.keys())[0]] is not None:
        raise ValueError('First folder value should be None.')

    originFolder = list(COMBINE_FOLDERS.keys())[0]
    addonFolder = list(COMBINE_FOLDERS.keys())[1]

    # Take `originFolder` as basis
    resDir = Path(BASE_FOLDER) / originFolder
    listOfFiles = list(resDir.glob('*.pkl'))
    listOfFiles = [file for file in listOfFiles if not file.stem.endswith('_metrics')]
    for file in listOfFiles:
        combName = Path(BASE_FOLDER) / addonFolder / file.name
        if combName.exists():
            print(f'Combining {file.name}...')
            dataOrigin = np.load(file, allow_pickle=True)
            dataAddOn = np.load(combName, allow_pickle=True)
            dataOut = copy.deepcopy(dataOrigin)
            if 'shatk' in dataOrigin.keys():
                for key in COMBINE_FOLDERS[addonFolder].keys():
                    newKey = COMBINE_FOLDERS[addonFolder][key]
                    dataOut['iSaved'][newKey] = dataAddOn['iSaved'][key]
                    for ii in range(len(dataOrigin['shatk'])):
                        dataOut['shatk'][ii][newKey] = dataAddOn['shatk'][ii][key]
                        dataOut['nhatk'][ii][newKey] = dataAddOn['nhatk'][ii][key]
            else:
                raise NotImplementedError('Combination for this type of data not implemented yet.')
            filenameOut = file.parent / (file.stem + '_comb.pkl')
            pickle.dump(dataOut, open(filenameOut, 'wb'))

if __name__ == '__main__':
    sys.exit(main())