"""
h5py 'extensions'.
"""
from typing import TypeVar

import h5py

def dset_extend1d(file: h5py.File, key: str, i: int, value: TypeVar("T")):
    """
    Dump and auto-extend a 1d extendible dataset.

    :param file: Opened HDF5 file.
    :param key: Path to the dataset.
    :param i: Index to which to write.
    :param value: Value to write at index ``i``.
    """

    dset = file[key]
    if dset.size <= i:
        dset.resize((i + 1,))
    dset[i] = value


def dump_with_atttrs(file: h5py.File, key: str, data: TypeVar("T"), **kwargs):
    """
    Write dataset and an optional number of attributes.
    The attributes are stored based on the name that is used for the option.

    :param file: Opened HDF5 file.
    :param key: Path to the dataset.
    :param data: Data to write.
    """

    file[key] = data
    for attr in kwargs:
        file[key].attrs[attr] = kwargs[attr]

