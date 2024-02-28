import re
from collections import defaultdict

import h5py
import numpy as np
from numpy.typing import ArrayLike
from scipy.ndimage import measurements


class Paths:
    """
    Keep track of stored data.
    """

    def __init__(self, file: h5py.File):
        """
        :param file: Opened HDF5 archive.
        """

        self.m_F = defaultdict(list)
        self.m_R = defaultdict(list)

        for root in file:
            if re.match(r"(.*)(n=)([0-9])", root):
                n = re.split(r"(.*)(n=)([0-9])", root)[3]
                for path in file[root]:
                    if re.match(r"(F_)([0-9])", path):
                        self.m_F[f"n={n}"].append(f"/{root}/{path}")
                        self.m_R[f"n={n}"].append(f"/{root}/{path.replace('F_', 'R_')}")

    def R(self, n: int, e: int) -> str:
        """
        Return path to the position of a certain experiment.

        :oaram n: Number of plates (``1, 2, ...``).
        :param e: Index of the experiment.
        :return: Path as string.
        """
        return self.m_R[f"n={n}"][e]

    def F(self, n: int, e: int) -> str:
        """
        Return path to the force of a certain experiment.

        :oaram n: Number of plates (``1, 2, ...``).
        :param e: Index of the experiment.
        :return: Path as string.
        """
        return self.m_F[f"n={n}"][e]

    def n_plates(self) -> int:
        """
        Return maximum number of plates.
        """
        ret = 0
        for key in self.m_R:
            ret = max(ret, int(key.split("=")[1]))
        return ret

    def n_exp(self, n: int) -> int:
        """
        Return number of experiments.

        :oaram n: Number of plates (``1, 2, ...``).
        """
        return len(self.m_R[f"n={n}"])


def identify_slip(
    t_F: ArrayLike, F: ArrayLike, t_R: ArrayLike, R: ArrayLike, delta_R: float = 0.02
) -> dict:
    """
    Identify slip events.

    This function first identifies slip events based on a threshold in the change in position,
    they occur during ``t_R[i0:i1]``.
    The corresponding force amplitude comes from subtracting
    the minimal force from the maximal force in the time window ``t_R[i0 - 1: i1 + 1]`` .
    The latter accounts from the time resolution being higher for force than from position.

    :param t_F: Time, corresponding to measured ``F``.
    :param F: Force.
    :param t_R: Time, corresponding to measured ``R``.
    :param R: Position.
    :return: Dictionary with indices of slip events in ``R`` and ``F``.
    """

    slip = np.diff(R, prepend=0) > delta_R
    labels, n = measurements.label(slip)

    ret_R = []
    ret_F = []
    sorter = []

    for label in np.arange(1, n + 1):

        i = np.argwhere(labels == label).ravel()
        i0 = i[0] - 1
        i1 = i[-1] + 1

        if i1 >= slip.size:
            continue

        j0 = np.argmax(t_R[i0 - 1] - t_F < 0)
        j1 = np.argmax(t_R[i1 + 1] - t_F < 0)

        if j0 == 0 or j1 == 0:
            continue

        sel = np.arange(j0, j1)
        j0 = sel[np.argmax(F[sel])]
        j1 = sel[np.argmin(F[sel])]

        ret_R.append([i0, i1])
        ret_F.append([j0, j1])
        sorter.append(i0)

    sorter = np.argsort(sorter)

    return dict(
        R=[ret_R[i] for i in sorter],
        F=[ret_F[i] for i in sorter],
    )


def read_parameters(string: str, convert: dict = {"n": int, "e": int}) -> dict:
    """
    Read parameters from a string: it is assumed that parameters are split by ``_`` or ``/``
    and that parameters are stored as ``name=value``.

    :param string: ``key=value`` separated by ``/`` or ``_``.
    :param convert: Type conversion for a selection of keys. E.g. ``{"id": int}``.
    :return: Parameters as dictionary.
    """

    part = re.split("_|/", string)

    ret = {}

    for i in part:
        if len(i.split("=")) > 1:
            key, value = i.split("=")
            ret[key] = value

    if convert:
        for key in convert:
            ret[key] = convert[key](ret[key])

    return ret


def colors(name: str) -> list:
    """
    Return color-cycle.

    :param name: ``"n"`` or ``"i"``.
    :return: List of colors.
    """

    if name == "i":
        c = (
            np.array(
                [
                    [152, 78, 163],
                    [152, 78, 163],
                    [228, 26, 28],
                    [55, 126, 184],
                    [77, 175, 74],
                    [255, 127, 0],
                ]
            )
            / 255
        )
    else:
        c = (
            np.array(
                [
                    # [152.0000, 78.0000, 163.0000],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [47.7044, 93.2453, 157.1853],
                    [83.0613, 133.3842, 127.2201],
                    [144.6349, 195.3938, 110.7479],
                    [253.0000, 188.0000, 66.0000],
                ]
            )
            / 255
        )

    return [i for i in c]


def markers(name: str) -> list:
    """
    Return marker-cycle.

    :param name: ``"n"`` or ``"i"``.
    :return: List of markers.
    """

    return [".", ".", ">", "^", "d", "s"]


def labels(name: str) -> list:
    """
    Return marker-cycle.

    :param name: ``"n"`` or ``"i"``.
    :return: List of markers.
    """

    return [rf"${name} = {i}$" for i in range(nplate() + 1)]


def nplate() -> int:
    """
    Maximum number of plates to consider.
    """
    return 5


def nlayer() -> int:
    """
    Maximum number of layers to consider.
    """
    return 2 * nplate() - 1
