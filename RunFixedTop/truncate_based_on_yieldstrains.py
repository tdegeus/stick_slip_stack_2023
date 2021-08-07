import argparse
import FrictionQPotFEM.UniformSingleLayer2d as model
import GMatElastoPlasticQPot.Cartesian2d as GMat
import GooseFEM as gf
import GooseHDF5 as g5
import h5py
import numpy as np
import os
import prrng
import QPot
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, default="input file")
parser.add_argument("output", type=str, default="output file")
args = parser.parse_args()

assert os.path.isfile(args.input)
assert not os.path.isfile(args.output)


def read_epsy(data, N):

    initstate = data["/cusp/epsy/initstate"][...]
    initseq = data["/cusp/epsy/initseq"][...]
    eps_offset = data["/cusp/epsy/eps_offset"][...]
    eps0 = data["/cusp/epsy/eps0"][...]
    k = data["/cusp/epsy/k"][...]
    nchunk = data["/cusp/epsy/nchunk"][...]

    generators = prrng.pcg32_array(initstate, initseq)

    epsy = generators.weibull([nchunk], k)
    epsy *= (2.0 * eps0)
    epsy += eps_offset
    epsy = np.cumsum(epsy, 1)

    return epsy


def checkbounds(system):

    shape = system.material_plastic().shape()

    for i in range(shape[0]):
        for j in range(shape[1]):
            m = system.material_plastic().refCusp([i, j])
            y = m.refQPotChunked()
            if not y.boundcheck_right(5):
                return False

    return True


with h5py.File(args.input, "r") as data:

    paths = list(g5.getdatasets(data))

    with h5py.File(args.input, "r") as data:

        system = model.System(
            data["/coor"][...],
            data["/conn"][...],
            data["/dofs"][...],
            data["/iip"][...],
            data["/elastic/elem"][...],
            data["/cusp/elem"][...])

        system.setMassMatrix(data["/rho"][...])
        system.setDampingMatrix(data["/damping/alpha"][...])

        system.setElastic(
            data["/elastic/K"][...],
            data["/elastic/G"][...])

        system.setPlastic(
            data["/cusp/K"][...],
            data["/cusp/G"][...],
            read_epsy(data, system.plastic().size))

        system.setDt(data["/run/dt"][...])

        for inc in tqdm.tqdm(data["/stored"][...]):

            u = data["/disp/{0:d}".format(inc)][...]
            system.setU(u)

            if not checkbounds(system):
                break

        ninc = inc


with h5py.File(args.input, "r") as data:

    paths = list(g5.getdatasets(data))

    paths.remove("/stored")
    paths.remove("/kick")
    paths.remove("/t")

    rm = data["/stored"][...]
    rm = rm[rm >= ninc]

    for i in rm:
        paths.remove("/disp/{0:d}".format(i))

    with h5py.File(args.output, 'w') as ret:

        g5.copydatasets(data, ret, paths)

        for key in ["stored", "kick"]:
            dset = ret.create_dataset(key, (ninc,), maxshape=(None,), dtype=int)
            dset[:] = data[key][...][:ninc]

        for key in ["t"]:
            dset = ret.create_dataset(key, (ninc,), maxshape=(None,))
            dset[:] = data[key][...][:ninc]


