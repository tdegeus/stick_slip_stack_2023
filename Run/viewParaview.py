import argparse
import os
import tqdm
import h5py
import numpy as np
import prrng
import GooseFEM as gf
import GMatElastoPlasticQPot.Cartesian2d as gmat
import FrictionQPotFEM.UniformMultiLayerIndividualDrive2d as model
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
    epsy *= (2.0 * eps0)
    epsy += eps_offset
    epsy = np.cumsum(epsy, 1)

    return epsy

for file in tqdm.tqdm(args.files):

    outbasename = "{0:s}_{1:s}".format(args.prefix, os.path.splitext(file)[0])

    with h5py.File(outbasename + ".hdf5", "w") as output:

        with h5py.File(file, "r") as data:

            layers = data["/layers/stored"][...]
            elemmap = []
            nodemap = []

            for layer in layers:
                elemmap += [data["/layers/{0:d}/elemmap".format(layer)][...]]
                nodemap += [data["/layers/{0:d}/nodemap".format(layer)][...]]

            system = model.System(
                data["/coor"][...],
                data["/conn"][...],
                data["/dofs"][...],
                data["/iip"][...],
                elemmap,
                nodemap,
                data["/layers/is_plastic"][...])

            system.setDriveStiffness(data["/drive/k"][...], data["/drive/symmetric"][...])
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

            dV = system.quad().AsTensor(2, system.quad().dV())

            output["/coor"] = system.coor()
            output["/conn"] = system.conn()

            series = xh.TimeSeries()

            for inc in tqdm.tqdm(data["/stored"][...]):

                system.layerSetUbar(
                    data["/drive/ubar/{0:d}".format(inc)][...],
                    data["/drive/drive"][...])

                u = data["/disp/{0:d}".format(inc)][...]
                system.setU(u)

                Sig = gmat.Sigd(np.average(system.Sig(), weights=dV, axis=1))

                output["/disp/{0:d}".format(inc)] = xh.as3d(u)
                output["/sigd/{0:d}".format(inc)] = Sig

                series.push_back(
                    xh.Unstructured(output, "/coor", "/conn", "Quadrilateral"),
                    xh.Attribute(output, "/disp/{0:d}".format(inc), "Node", name="Displacement"),
                    xh.Attribute(output, "/sigd/{0:d}".format(inc), "Cell", name="Stress"))

            xh.write(series, outbasename + ".xdmf")



