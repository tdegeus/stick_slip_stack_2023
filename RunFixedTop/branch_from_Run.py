import argparse
import os

import GooseHDF5 as g5
import h5py
from setuptools_scm import get_version

version = get_version(root=os.path.join(os.path.dirname(__file__), ".."))

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, default="input file")
parser.add_argument("output", type=str, default="output file")
parser.add_argument("-i", "--inc", type=int, default="increment at which to branch")
args = parser.parse_args()

assert os.path.isfile(args.input)
assert not os.path.isfile(args.output)


with h5py.File(args.input, "r") as data:

    paths = list(g5.getdatasets(data))

    paths.remove("/stored")
    paths.remove("/t")
    paths.remove("/drive/k")
    paths.remove("/drive/symmetric")
    paths.remove("/drive/drive")
    paths.remove("/drive/delta_gamma")
    paths.remove("/drive/height")

    for i in data["/stored"][...]:
        paths.remove(f"/disp/{i:d}")

    with h5py.File(args.output, "w") as ret:

        g5.copydatasets(data, ret, paths)

        dset = ret.create_dataset("stored", (1,), maxshape=(None,), dtype=int)
        dset[0] = 0

        dset = ret.create_dataset("kick", (1,), maxshape=(None,), dtype=int)
        dset[0] = 0

        dset = ret.create_dataset("t", (1,), maxshape=(None,), dtype=float)
        dset[0] = data["t"][args.inc]

        ret[f"/disp/{0:d}"] = data[f"/disp/{args.inc:d}"][...]

        key = "/meta/RunFixedBoundary/generate.py"
        ret[key] = version
        ret[key].attrs["desc"] = "Version when generating"

        key = "/meta/RunFixedBoundary/branch/inc"
        ret[key] = args.inc
        ret[key].attrs["desc"] = "Increment from which this simulation is branched"
