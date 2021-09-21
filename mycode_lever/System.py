import FrictionQPotFEM.UniformMultiLayerIndividualDrive2d as model
import h5py
import numpy as np
import prrng


def read_epsy(file: h5py.File) -> np.ndarray:
    """
    Regenerate yield strain sequence per plastic element.
    The output shape is given by the stored ``initstate``.

    :param file: Opened simulation archive.
    """

    initstate = file["/cusp/epsy/initstate"][...]
    initseq = file["/cusp/epsy/initseq"][...]
    eps_offset = file["/cusp/epsy/eps_offset"][...]
    eps0 = file["/cusp/epsy/eps0"][...]
    k = file["/cusp/epsy/k"][...]
    nchunk = file["/cusp/epsy/nchunk"][...]

    generators = prrng.pcg32_array(initstate, initseq)

    epsy = generators.weibull([nchunk], k)
    epsy *= 2.0 * eps0
    epsy += eps_offset
    epsy = np.cumsum(epsy, 1)

    return epsy


def init(file: h5py.File) -> model.System:
    """
    Initialise system from file.

    :param file: Open simulation HDF5 archive (read-only).
    :return: The initialised system.
    """

    layers = file["/layers/stored"][...]

    system = model.System(
        file["coor"][...],
        file["conn"][...],
        file["dofs"][...],
        file["iip"][...],
        [file[f"/layers/{layer:d}/elemmap"][...] for layer in layers],
        [file[f"/layers/{layer:d}/nodemap"][...] for layer in layers],
        file["/layers/is_plastic"][...],
    )

    system.setMassMatrix(file["rho"][...])
    system.setDampingMatrix(file["/damping/alpha"][...])

    system.setElastic(file["/elastic/K"][...], file["/elastic/G"][...])
    system.setPlastic(file["/cusp/K"][...], file["/cusp/G"][...], read_epsy(file))

    system.setDt(file["/run/dt"][...])

    system.layerSetTargetActive(file["/drive/drive"][...])
    system.layerSetDriveStiffness(file["/drive/k"][...], file["/drive/symmetric"][...])

    return system
