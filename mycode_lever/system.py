import FrictionQPotFEM.UniformMultiLayerIndividualDrive2d as model
import h5py
import numpy as np
import prrng


def read_epsy(data: h5py.File) -> np.ndarray:
    """
    Regenerate yield strain sequence per plastic element.
    The output shape is given by the stored ``initstate``.

    :param data: Opened simulation archive.
    """

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


def initsystem(data: h5py.File) -> model.System:
    """
    Restore system.

    :param data: Opened simulation archive.
    """

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

    system.setDt(data["/run/dt"][...])
    system.setMassMatrix(data["/rho"][...])
    system.setDampingMatrix(data["/damping/alpha"][...])

    system.setElastic(data["/elastic/K"][...], data["/elastic/G"][...])
    system.setPlastic(
        data["/cusp/K"][...],
        data["/cusp/G"][...],
        read_epsy(data),
    )

    system.layerSetTargetActive(data["/drive/drive"][...])
    system.layerSetDriveStiffness(data["/drive/k"][...], data["/drive/symmetric"][...])

    return system
