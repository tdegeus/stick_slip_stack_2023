import argparse
import inspect
import os
import re
import sys
import textwrap
from collections import defaultdict

import GooseHDF5 as g5
import h5py
import numpy as np
import prrng
from numpy.typing import ArrayLike

from . import storage
from . import tag
from ._version import version

entry_points = dict(
    cli_compare="System_Compare",
)


def replace_entry_point(doc):
    """
    Replace ":py:func:`...`" with the relevant entry_point name
    """
    for ep in entry_points:
        doc = doc.replace(fr":py:func:`{ep:s}`", entry_points[ep])
    return doc


def dependencies(model) -> list[str]:
    """
    Return list with version strings.
    Compared to model.System.version_dependencies() this added the version of prrng.

    :param model: Relevant FrictionQPotFEM module.
    :return: List of version strings.
    """
    return sorted(list(model.version_dependencies()) + ["prrng=" + prrng.version()])


def interpret_filename(filename):
    """
    Split filename in useful information.
    """

    part = re.split("_|/", os.path.splitext(filename)[0])
    info = {}

    for i in part:
        key, value = i.split("=")
        info[key] = value

    for key in info:
        if key in ["kplate"]:
            info[key] = float(info[key])
        else:
            info[key] = int(info[key])

    return info


def compare(a: h5py.File, b: h5py.File):
    """
    Compare two file: will be replaced by :py:func:`GooseHDF5.compare`.
    """

    paths_a = list(g5.getdatasets(a, fold=["/disp", "/drive/ubar"]))
    paths_b = list(g5.getdatasets(b, fold=["/disp", "/drive/ubar"]))
    paths_a = [p for p in paths_a if p[-3:] != "..."]
    paths_b = [p for p in paths_b if p[-3:] != "..."]
    ret = defaultdict(list)

    not_in_b = [str(i) for i in np.setdiff1d(paths_a, paths_b)]
    not_in_a = [str(i) for i in np.setdiff1d(paths_b, paths_a)]
    inboth = [str(i) for i in np.intersect1d(paths_a, paths_b)]

    for path in not_in_a:
        ret["<-"].append(path)

    for path in not_in_b:
        ret["->"].append(path)

    for path in inboth:
        if not g5.equal(a, b, path):
            ret["!="].append(path)

    return ret


def cli_compare(cli_args=None):
    """
    Compare input files.
    """

    class MyFmt(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=replace_entry_point(doc))

    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("file_a", type=str, help="Simulation file")
    parser.add_argument("file_b", type=str, help="Simulation file")

    if cli_args is None:
        args = parser.parse_args(sys.argv[1:])
    else:
        args = parser.parse_args([str(arg) for arg in cli_args])

    assert os.path.isfile(os.path.realpath(args.file_a))
    assert os.path.isfile(os.path.realpath(args.file_b))

    with h5py.File(args.file_a, "r") as a:
        with h5py.File(args.file_b, "r") as b:
            ret = compare(a, b)

    for key in ret:
        if key in ["=="]:
            continue
        for path in ret[key]:
            print(key, path)

    if cli_args is not None:
        return ret


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
        initstate=file["/cusp/epsy/initstate"][...],
        initseq=file["/cusp/epsy/initseq"][...],
        eps_offset=file["/cusp/epsy/eps_offset"][...],
        eps0=file["/cusp/epsy/eps0"][...],
        k=file["/cusp/epsy/k"][...],
        nchunk=file["/cusp/epsy/nchunk"][...],
    )


def init(file: h5py.File, model):
    """
    Initialise system from file.

    :param file: Open simulation HDF5 archive (read-only).
    :param model: Relevant FrictionQPotFEM module.
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


def run(config: str, progname: str, model, init_function, filepath: str, dev: bool):
    """
    Run the simulation.

    :param config: Name of the configuration ("FixedLever", or "FreeLever")
    :param progname: Name of command-line tool running the function.
    :param model: Relevant FrictionQPotFEM module.
    :param init_function: Initialisation function.
    :param filepath: Name of the input/output file (appended).
    :param dev: Allow uncommitted changes.
    """

    basename = os.path.basename(filepath)

    with h5py.File(filepath, "a") as file:

        system = init_function(file)

        if config == "FreeLever":
            system.setLeverProperties(file["/drive/H"][...], file["/drive/height"][...])

        # check version compatibility

        assert dev or not tag.has_uncommited(version)
        assert dev or not tag.any_has_uncommited(dependencies(model))

        if f"/meta/{config}/{progname}" not in file:
            meta = file.create_group(f"/meta/{config}/{progname}")
            meta.attrs["version"] = version
            meta.attrs["dependencies"] = dependencies(model)
        else:
            meta = file[f"/meta/{config}/{progname}"]

        if "completed" in meta:
            print("Marked completed, skipping")
            return 1

        assert tag.equal(version, meta.attrs["version"])
        assert tag.all_equal(dependencies(model), meta.attrs["dependencies"])

        # restore or initialise the system / output

        if "/stored" in file:

            inc = int(file["/stored"][-1])
            system.setT(file["/t"][inc])
            system.setU(file[f"/disp/{inc:d}"][...])
            system.layerSetTargetUbar(file[f"/drive/ubar/{inc:d}"][...])
            print(f'"{basename}": Loading, inc = {inc:d}')

        else:

            inc = int(0)
            desc = '(end of increment). One entry per item in "/stored".'

            storage.dset_extendible1d(
                file=file,
                key="/stored",
                dtype=np.uint64,
                value=inc,
                desc="List of stored increments.",
            )

            storage.dset_extendible1d(
                file=file,
                key="/t",
                dtype=np.float64,
                value=system.t(),
                desc=f"Time {desc}",
            )

            file[f"/disp/{inc}"] = system.u()
            file["/disp"].attrs["desc"] = f"Displacement {desc}"

            file[f"/drive/ubar/{inc}"] = system.layerTargetUbar()
            file["/drive/ubar"].attrs["desc"] = f"Loading frame position per layer {desc}"

            if config == "FreeLever":

                storage.dset_extendible1d(
                    file=file,
                    key="/drive/lever/target",
                    dtype=float,
                    value=system.leverTarget(),
                    desc=f"Target position of the lever {desc}",
                )

                storage.dset_extendible1d(
                    file=file,
                    key="/drive/lever/position",
                    dtype=float,
                    value=system.leverPosition(),
                    desc=f"Actual position of the lever {desc}",
                )

        # run

        if config == "FixedLever":
            height = file["/drive/height"][...]
            delta_gamma = file["/drive/delta_gamma"][...]
        else:
            height = file["/drive/H"][...]
            delta_gamma = np.cumsum(file["/drive/delta_gamma"][...])

        assert np.isclose(delta_gamma[0], 0.0)
        inc += 1

        for dgamma in delta_gamma[inc:]:

            if config == "FixedLever":
                system.layerTagetUbar_addAffineSimpleShear(dgamma, height)
            else:
                system.setLeverTarget(height * dgamma)  # dgamma == total strain, see above

            niter = system.minimise()
            if not system.boundcheck_right(5):
                break
            print(f'"{basename}": inc = {inc:8d}, niter = {niter:8d}')

            storage.dset_extend1d(file, "/stored", inc, inc)
            storage.dset_extend1d(file, "/t", inc, system.t())
            file[f"/disp/{inc:d}"] = system.u()
            file[f"/drive/ubar/{inc:d}"] = system.layerTargetUbar()

            if config == "FreeLever":
                storage.dset_extend1d(file, "/drive/lever/target", inc, system.leverTarget())
                storage.dset_extend1d(file, "/drive/lever/position", inc, system.leverPosition())

            inc += 1

        print(f'"{basename}": completed')
        meta.attrs["completed"] = 1
