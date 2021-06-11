import subprocess
import h5py
import os
import tqdm
from shelephant.yaml import dump

def is_completed(file):
    with h5py.File(file, 'r') as data:
        if '/meta/Run/completed' in data:
            return data['/meta/Run/completed'][...]
    return False

files = sorted(list(filter(None, subprocess.check_output(
    "find . -maxdepth 1 -iname 'id*.h5'", shell=True).decode('utf-8').split('\n'))))

sims = [os.path.relpath(file) for file in tqdm.tqdm(files) if is_completed(file)]

dump('completed.yaml', sims)
