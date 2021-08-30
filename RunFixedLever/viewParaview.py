import argparse
import FrictionQPotFEM.UniformMultiLayerIndividualDrive2d as model
import GMatElastoPlasticQPot.Cartesian2d as GMat
import GooseFEM
import h5py
import numpy as np
import os
import prrng
import tqdm
import XDMFWrite_h5py as xh

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--prefix", type=str, default="paraview")
parser.add_argument("files", type=str, nargs="*")
args = parser.parse_args()
assert np.all([os.path.isfile(os.path.realpath(file)) for file in args.files])
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


def initsystem(data):

    layers = data["/layers/stored"][...]

    system = model.System(
        data["/coor"][...],
        data["/conn"][...],
        data["/dofs"][...],
        data["/iip"][...],
        [data[f"/layers/{layer:d}/elemmap"][...] for layer in layers],
        [data[f"/layers/{layer:d}/nodemap"][...] for layer in layers],
        data["/layers/is_plastic"][...],
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
    system.layerSetTargetActive(data["/drive/drive"][...])
    system.layerSetDriveStiffness(data["/drive/k"][...], data["/drive/symmetric"][...])

    return system


for file in tqdm.tqdm(args.files):

    outbasename = f"{args.prefix:s}_{os.path.splitext(file)[0]:s}"

    with h5py.File(outbasename + ".hdf5", "w") as output:

        with h5py.File(file, "r") as data:

            system = initsystem(data)
            dV = system.quad().AsTensor(2, system.quad().dV())
            sig0 = data["/meta/normalisation/sig"][...]
            eps0 = data["/meta/normalisation/eps"][...]

            output["/coor"] = system.coor()
            output["/conn"] = system.conn()

            series = xh.TimeSeries()

            for inc in tqdm.tqdm(data["/stored"][...]):

                system.layerSetTargetUbar(data[f"/drive/ubar/{inc:d}"][...])

                u = data[f"/disp/{inc:d}"][...]
                system.setU(u)

                Sig = GMat.Sigd(np.average(system.Sig() / sig0, weights=dV, axis=1))
                Epsp = np.zeros_like(Sig)
                Epsp[system.plastic()] = np.mean(system.plastic_Epsp() / eps0, axis=1)

                output[f"/disp/{inc:d}"] = xh.as3d(u)
                output[f"/sigd/{inc:d}"] = Sig
                output[f"/epsp/{inc:d}"] = Epsp

                series.push_back(
                    xh.Unstructured(output, "/coor", "/conn", "Quadrilateral"),
                    xh.Attribute(output, f"/disp/{inc:d}", "Node", name="Displacement"),
                    xh.Attribute(output, f"/sigd/{inc:d}", "Cell", name="Stress"),
                    xh.Attribute(
                        output, f"/epsp/{inc:d}", "Cell", name="Plastic strain"
                    ),
                )

            xh.write(series, outbasename + ".xdmf")
