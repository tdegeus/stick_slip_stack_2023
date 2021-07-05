import argparse
import h5py
import numpy as np
import os
import tqdm

from shelephant.yaml import dump

def is_completed(file):
    with h5py.File(file, 'r') as data:
        if '/meta/Run/completed' in data:
            return data['/meta/Run/completed'][...]
    return False

parser = argparse.ArgumentParser()
parser.add_argument('files', nargs='*', type=str)
args = parser.parse_args()
assert np.all([os.path.isfile(file) for file in args.files])

sims = [os.path.relpath(file) for file in tqdm.tqdm(args.files) if is_completed(file)]

dump('completed.yaml', sims)
