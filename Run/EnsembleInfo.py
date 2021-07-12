from setuptools_scm import get_version
import argparse
import FrictionQPotFEM.UniformMultiLayerIndividualDrive2d as model
import GMatElastoPlasticQPot.Cartesian2d as GMat
import GooseFEM
import h5py
import numpy as np
import os
import prrng
import tqdm

basename = os.path.splitext(os.path.basename(__file__))[0]

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", type=str, default=basename + ".h5")
parser.add_argument("files", type=str, nargs="*")
args = parser.parse_args()
assert np.all([os.path.isfile(os.path.realpath(file)) for file in args.files])
assert len(args.files) > 0
filenames = [os.path.basename(file) for file in args.files]


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


def initsystem(data):

    layers = data["/layers/stored"][...]

    system = model.System(
        data["/coor"][...],
        data["/conn"][...],
        data["/dofs"][...],
        data["/iip"][...],
        [data["/layers/{0:d}/elemmap".format(layer)][...] for layer in layers],
        [data["/layers/{0:d}/nodemap".format(layer)][...] for layer in layers],
        data["/layers/is_plastic"][...])

    system.setDriveStiffness(data["/drive/k"][...], data["/drive/symmetric"][...])
    system.setMassMatrix(data["/rho"][...])
    system.setDampingMatrix(data["/damping/alpha"][...])
    system.setElastic(data["/elastic/K"][...], data["/elastic/G"][...])
    system.setPlastic(data["/cusp/K"][...], data["/cusp/G"][...], read_epsy(data, system.plastic().size))
    system.setDt(data["/run/dt"][...])

    return system


for ifile, file in enumerate(tqdm.tqdm(args.files)):

    with h5py.File(file, "r") as data:

        system = initsystem(data)
        nlayer = system.nlayer()
        dV = system.quad().AsTensor(2, system.quad().dV())

        if ifile == 0:
            N = data["/meta/normalisation/N"][...]
            eps0 = data["/meta/normalisation/eps"][...]
            sig0 = data["/meta/normalisation/sig"][...]
            dt = data["/run/dt"][...]
            kdrive = data["/drive/k"][...]
        else:
            assert np.isclose(N, data["/meta/normalisation/N"][...])
            assert np.isclose(eps0, data["/meta/normalisation/eps"][...])
            assert np.isclose(sig0, data["/meta/normalisation/sig"][...])
            assert np.isclose(dt, data["/run/dt"][...])
            assert np.isclose(kdrive, data["/drive/k"][...])

        incs = data["/stored"][...]
        ninc = incs.size
        idx_n = system.plastic_CurrentIndex().astype(int)[:, 0].reshape(-1, N)

        is_plastic = data["/layers/is_plastic"][...]
        Drive = data["/drive/drive"][...]
        Drive_x =  np.argwhere(Drive[:, 0]).ravel()
        Height = data["/drive/height"][...]
        Dgamma = data["/drive/delta_gamma"][...][incs]

        Strain = np.empty((ninc), dtype=float)
        Stress = np.empty((ninc), dtype=float)
        Strain_layers = np.empty((ninc, nlayer), dtype=float)
        Stress_layers = np.empty((ninc, nlayer), dtype=float)
        S_layers = np.zeros((ninc, nlayer), dtype=int)
        A_layers = np.zeros((ninc, nlayer), dtype=int)
        Drive_Ux = np.zeros((ninc, Drive_x.size), dtype=float)
        Drive_Fx = np.zeros((ninc, Drive_x.size), dtype=float)

        for inc in tqdm.tqdm(incs):

            ubar = data["/drive/ubar/{0:d}".format(inc)][...]
            system.layerSetTargetUbar(ubar, Drive)

            u = data["/disp/{0:d}".format(inc)][...]
            system.setU(u)

            Drive_Ux[inc, :] = ubar[Drive_x, 0] / Height[Drive_x]
            Drive_Fx[inc, :] = system.layerFdrive()[Drive_x, 0] / kdrive
            Sig = system.Sig() / sig0
            Eps = system.Eps() / eps0
            idx = system.plastic_CurrentIndex().astype(int)[:, 0].reshape(-1, N)

            for i in range(nlayer):
                e = system.layerElements(i)
                E = np.average(Eps[e, ...], weights=dV[e, ...], axis=(0, 1))
                S = np.average(Sig[e, ...], weights=dV[e, ...], axis=(0, 1))
                Strain_layers[inc, i] = GMat.Epsd(E)
                Stress_layers[inc, i] = GMat.Sigd(S)

            S_layers[inc, is_plastic] = np.sum(idx - idx_n, axis=1)
            A_layers[inc, is_plastic] = np.sum(idx != idx_n, axis=1)
            Strain[inc] = GMat.Epsd(np.average(Eps, weights=dV, axis=(0, 1)))
            Stress[inc] = GMat.Sigd(np.average(Sig, weights=dV, axis=(0, 1)))

            idx_n = np.array(idx, copy=True)

    with h5py.File(args.output, "a" if ifile > 0 else "w") as output:

        key = "/file/{0:s}/drive/drive".format(os.path.normpath(file))
        output[key] = Drive
        output[key].attrs["desc"] = "Drive per layer and direction"

        key = "/file/{0:s}/drive/height".format(os.path.normpath(file))
        output[key] = Height
        output[key].attrs["desc"] = "Height of the loading frame of each layer"

        key = "/file/{0:s}/drive/delta_gamma".format(os.path.normpath(file))
        output[key] = Dgamma / eps0
        output[key].attrs["desc"] = "Applied shear [ninc], in units of eps0"

        key = "/file/{0:s}/drive/ux".format(os.path.normpath(file))
        output[key] = Drive_Ux
        output[key].attrs["desc"] = \
            "Drive position in x-direction on driven layers divided by layer height [ninc, ndrive]"

        key = "/file/{0:s}/drive/fx".format(os.path.normpath(file))
        output[key] = Drive_Fx
        output[key].attrs["desc"] = "Drive force in x-direction on driven layers [ninc, nlayer]"

        key = "/file/{0:s}/macroscopic/eps".format(os.path.normpath(file))
        output[key] = Strain
        output[key].attrs["desc"] = "Macroscopic strain per increment [ninc], in units of eps0"

        key = "/file/{0:s}/macroscopic/sig".format(os.path.normpath(file))
        output[key] = Stress
        output[key].attrs["desc"] = "Macroscopic stress per increment [ninc], in units of sig0"

        key = "/file/{0:s}/layers/eps".format(os.path.normpath(file))
        output[key] = Strain_layers
        output[key].attrs["desc"] = "Average strain per layer [ninc, nlayer], in units of eps0"

        key = "/file/{0:s}/layers/sig".format(os.path.normpath(file))
        output[key] = Stress_layers
        output[key].attrs["desc"] = "Average stress per layer [ninc, nlayer], in units of sig0"

        key = "/file/{0:s}/layers/S".format(os.path.normpath(file))
        output[key] = S_layers
        output[key].attrs["desc"] = "Total number of yield events per layer [ninc, nlayer]"

        key = "/file/{0:s}/layers/A".format(os.path.normpath(file))
        output[key] = A_layers
        output[key].attrs["desc"] = "Total number of blocks that yields per layer [ninc, nlayer]"

with h5py.File(args.output, "a") as output:

    key = "/normalisation/N"
    output[key] = N
    output[key].attrs["desc"] = "Number of blocks along each plastic layer"

    key = "/normalisation/kdrive"
    output[key] = kdrive
    output[key].attrs["desc"] = "Driving spring stiffness"

    key = "/normalisation/sig0"
    output[key] = sig0
    output[key].attrs["desc"] = "Unit of stress"

    key = "/normalisation/eps0"
    output[key] = eps0
    output[key].attrs["desc"] = "Unit of strain"

    key = "/normalisation/dt"
    output[key] = dt
    output[key].attrs["desc"] = "Time step"

    key = "/meta/Run/EnsembleInfo.py"
    output[key] = get_version(root=os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
    output[key].attrs["desc"] = "Version at which this file was created"
