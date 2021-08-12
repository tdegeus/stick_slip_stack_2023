import argparse
import h5py
import numpy as np
import os
import shelephant.yaml
import tqdm

basename = os.path.split(os.path.dirname(os.path.realpath(__file__)))[1]

def is_completed(file):
    with h5py.File(file, "r") as data:
        if f"/meta/{basename}/completed" in data:
            return data[f"/meta/{basename}/completed"][...]
    return False

parser = argparse.ArgumentParser()
parser.add_argument("files", nargs="*", type=str)
args = parser.parse_args()
assert np.all([os.path.isfile(file) for file in args.files])

sims = [os.path.relpath(file) for file in tqdm.tqdm(args.files) if is_completed(file)]

shelephant.yaml.dump("completed.yaml", sims)
