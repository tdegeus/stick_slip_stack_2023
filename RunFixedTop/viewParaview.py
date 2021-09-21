import argparse
import os

import FrictionQPotFEM.UniformSingleLayer2d as model
import GMatElastoPlasticQPot.Cartesian2d as GMat
import GooseFEM  # noqa: F401
import h5py
import numpy as np
import prrng
import tqdm
import XDMFWrite_h5py as xh

parser = argparse.ArgumentParser()
parser.add_argument("files", type=str, nargs="*")
parser.add_argument("-p", "--prefix", type=str, default="paraview")
args = parser.parse_args()
assert np.all([os.path.isfile(file) for file in args.files])
assert len(args.files) > 0


def read_epsy(data, N):

    initstate = data["/cusp/epsy/initstate"][...]
    initseq = data["/cusp/epsy/initseq"][...]
    eps_offset = data["/cusp/epsy/eps_offset"][...]
    eps0 = data["/cusp/epsy/eps0"][...]
    k = data["/cusp/epsy/k"][...]
    nchunk = data["/cusp/epsy/nchunk"][...]

    generators = prrng.pcg32_array(initstate, initseq)

    epsy = generators.weibull([nchunk], k)
    epsy *= 2.0 * eps0
    epsy += eps_offset
    epsy = np.cumsum(epsy, 1)

    return epsy


for file in tqdm.tqdm(args.files):

    outbasename = f"{args.prefix:s}_{os.path.splitext(file)[0]:s}"

    with h5py.File(outbasename + ".hdf5", "w") as output:

        with h5py.File(file, "r") as data:

            system = model.System(
                data["/coor"][...],
                data["/conn"][...],
                data["/dofs"][...],
                data["/iip"][...],
                data["/elastic/elem"][...],
                data["/cusp/elem"][...],
            )

            system.setMassMatrix(data["/rho"][...])
            system.setDampingMatrix(data["/damping/alpha"][...])

            system.setElastic(data["/elastic/K"][...], data["/elastic/G"][...])

            system.setPlastic(
                data["/cusp/K"][...],
                data["/cusp/G"][...],
                read_epsy(data, system.plastic().size),
            )

            system.setDt(data["/run/dt"][...])

            dV = system.quad().AsTensor(2, system.quad().dV())
            sig0 = data["/meta/normalisation/sig"][...]

            output["/coor"] = system.coor()
            output["/conn"] = system.conn()

            series = xh.TimeSeries()

            for inc in tqdm.tqdm(data["/stored"][...]):

                u = data[f"/disp/{inc:d}"][...]
                system.setU(u)

                Sig = GMat.Sigd(np.average(system.Sig() / sig0, weights=dV, axis=1))

                output[f"/disp/{inc:d}"] = xh.as3d(u)
                output[f"/sigd/{inc:d}"] = Sig

                series.push_back(
                    xh.Unstructured(output, "/coor", "/conn", "Quadrilateral"),
                    xh.Attribute(output, f"/disp/{inc:d}", "Node", name="Displacement"),
                    xh.Attribute(output, f"/sigd/{inc:d}", "Cell", name="Stress"),
                )

            xh.write(series, outbasename + ".xdmf")
