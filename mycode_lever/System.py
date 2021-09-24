import FrictionQPotFEM.UniformMultiLayerIndividualDrive2d as model
import h5py
import numpy as np
import prrng
from numpy.typing import ArrayLike

def get_epsy(initstate, initseq, eps_offset, eps0, k, nchunk):
    """
    Get yield strain sequence.

    :param initstate: State of the random number generator.
    :param initseq: State of the random number generator.
    :param eps_offset: Offset to apply to each drawn random number.
    :param eps0: Typical yield strain: the distance between two cusps is twice this.
    :param k: Shape parameter of the Weibull distribution.
    :param nchunk: Chunk size.
    """

    generators = prrng.pcg32_array(initstate, initseq)

    epsy = generators.weibull([nchunk], k)
    epsy *= 2.0 * eps0
    epsy += eps_offset
    epsy = np.cumsum(epsy, 1)

    return epsy


def read_epsy(file: h5py.File) -> np.ndarray:
    """
    Regenerate yield strain sequence per plastic element.
    The output shape is given by the stored ``initstate``.

    :param file: Opened simulation archive.
    """

    return get_epsy(
        initstate = file["/cusp/epsy/initstate"][...],
        initseq = file["/cusp/epsy/initseq"][...],
        eps_offset = file["/cusp/epsy/eps_offset"][...],
        eps0 = file["/cusp/epsy/eps0"][...],
        k = file["/cusp/epsy/k"][...],
        nchunk = file["/cusp/epsy/nchunk"][...],
    )


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


def steadystate(epsd: ArrayLike, sigd: ArrayLike, **kwargs):
    """
    Estimate the first increment of the steady-state, with additional constraints:
    -   Skip at least two increments.
    -   Start with elastic loading.

    :param epsd: Strain history [ninc].
    :param sigd: Stress history [ninc].
    :return: Increment number.
    """

    K = np.empty_like(sigd)
    K[0] = np.inf
    K[1:] = (sigd[1:] - sigd[0]) / (epsd[1:] - epsd[0])

    steadystate = max(2, np.argmax(K <= 0.95 * K[1]))
    return steadystate
