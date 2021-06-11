import argparse
import os
import tqdm
import h5py
import numpy as np
import prrng
import GooseFEM as gf
import GMatElastoPlasticQPot.Cartesian2d as gmat
import FrictionQPotFEM.UniformMultiLayerIndividualDrive2d as model

parser = argparse.ArgumentParser()
parser.add_argument("files", type=str, nargs="*")
parser.add_argument("-o", "--output", type=str, default="EnsembleInfo.h5")
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

with h5py.File(args.output, "w") as output:

    for file in tqdm.tqdm(args.files):

        with h5py.File(file, "r") as data:

            layers = data["/layers/stored"][...]
            nlayers = layers.size
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

            K_elas = data["/elastic/K"][...]
            G_elas = data["/elastic/G"][...]
            K_plas = data["/cusp/K"][...]
            G_plas = data["/cusp/G"][...]
            K0 = K_elas.ravel()[0]
            G0 = G_elas.ravel()[0]
            eps0 = data["/cusp/epsy/eps0"][...]
            sig0 = 2.0 * G0 * eps0 # because of definition of the equivalent strains

            assert np.allclose(K0, K_elas)
            assert np.allclose(G0, G_elas)
            assert np.allclose(K0, K_plas)
            assert np.allclose(G0, G_plas)

            system.setDriveStiffness(data["/drive/k"][...], data["/drive/symmetric"][...])
            system.setMassMatrix(data["/rho"][...])
            system.setDampingMatrix(data["/damping/alpha"][...])
            system.setElastic(K_elas, G_elas)
            system.setPlastic(K_plas, G_plas, read_epsy(data, system.plastic().size))
            system.setDt(data["/run/dt"][...])

            dV = system.quad().AsTensor(2, system.quad().dV())

            incs = data["/stored"][...]
            ninc = incs.size
            idx_n = system.plastic_CurrentIndex().astype(int).reshape(nlayers, -1)

            Strain = np.empty((ninc), dtype=float)
            Stress_layers = np.empty((ninc, nlayers), dtype=float)
            S_layers = np.empty((ninc, nlayers), dtype=int)
            A_layers = np.empty((ninc, nlayers), dtype=int)

            for inc in tqdm.tqdm(incs):

                system.layerSetUbar(
                    data["/drive/ubar/{0:d}".format(inc)][...],
                    data["/drive/drive"][...])

                u = data["/disp/{0:d}".format(inc)][...]
                system.setU(u)

                Sig = system.Sig() / sig0
                Eps = system.Eps() / eps0
                idx = system.plastic_CurrentIndex().astype(int).reshape(nlayers, -1)

                for i in range(nlayers):
                    Stress_layers[inc, i] = gmat.Sigd(np.average(Sig[elemmap[i], ...], weights=dV[elemmap[i], ...], axis=(0, 1)))

                S_layers[inc, :] = np.sum(idx - idx_n, axis=1)
                A_layers[inc, :] = np.sum(idx != idx_n, axis=1)
                Strain[inc] = gmat.Epsd(np.average(Eps, weights=dV, axis=(0, 1)))

                idx_n = np.array(idx, copy=True)

        key = "/raw/{0:s}/macroscopic/eps".format(os.path.normpath(file))
        output[key] = Strain
        output[key].attrs["desc"] = "Macroscopic strain per increment [ninc], in units of eps0"

        key = "/raw/{0:s}/layers/sig".format(os.path.normpath(file))
        output[key] = Stress_layers
        output[key].attrs["desc"] = "Average stress per layer [ninc, nlayers], in units of sig0"

        key = "/raw/{0:s}/layers/S".format(os.path.normpath(file))
        output[key] = S_layers
        output[key].attrs["desc"] = "Total number of yield events per layer [ninc, nlayers]"

        key = "/raw/{0:s}/layers/A".format(os.path.normpath(file))
        output[key] = A_layers
        output[key].attrs["desc"] = "Total number of blocks that yields per layer [ninc, nlayers]"
