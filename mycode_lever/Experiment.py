import re
from collections import defaultdict

import cppcolormap as cm
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


def colors() -> list:
    """
    Return color-cycle.

    :return: List of colors.
    """

    return [cm.tue()[0], cm.tue()[2], cm.tue()[4], cm.tue()[6], cm.tue()[8], cm.tue()[10]]
