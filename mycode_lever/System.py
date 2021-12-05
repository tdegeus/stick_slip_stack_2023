import argparse
import inspect
import itertools
import os
import re
import sys
import textwrap
from typing import Union

import click
import cppcolormap as cm
import GMatElastoPlasticQPot.Cartesian2d as GMat
import GooseFEM
import GooseHDF5 as g5
import GooseMPL as gplt
import h5py
import matplotlib.pyplot as plt
import numpy as np
import prrng
import tqdm
import yaml
from numpy.typing import ArrayLike

from . import mesh
from . import slurm
from . import storage
from . import tag
from ._version import version

plt.style.use(["goose", "goose-latex"])

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
    parser.add_argument("-d", "--datasets", action="store_true", help="Don't compare attributes.")
    parser.add_argument("file_a", type=str, help="Simulation file")
    parser.add_argument("file_b", type=str, help="Simulation file")

    if cli_args is None:
        args = parser.parse_args(sys.argv[1:])
    else:
        args = parser.parse_args([str(arg) for arg in cli_args])

    assert os.path.isfile(args.file_a)
    assert os.path.isfile(args.file_b)

    with h5py.File(args.file_a, "r") as a:
        with h5py.File(args.file_b, "r") as b:
            paths_a = list(g5.getdatasets(a, fold=["/disp", "/drive/ubar"]))
            paths_b = list(g5.getdatasets(b, fold=["/disp", "/drive/ubar"]))
            paths_a = [p for p in paths_a if p[-3:] != "..."]
            paths_b = [p for p in paths_b if p[-3:] != "..."]
            ret = g5.compare(a, b, paths_a, paths_b, attrs=not args.datasets)

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
    steadystate = K <= 0.95 * K[1]

    if not np.any(steadystate):
        return sys.maxsize

    return max(2, np.argmax(steadystate))


def generate(
    config: str,
    progname: str,
    filename: str,
    N: int,
    nplates: int,
    seed: int = 0,
    k_drive: float = 1e-3,
    symmetric: bool = True,
    test_mode: bool = False,
):
    """
    Generate an input file.

    :param config: Name of the configuration ("FixedLever", or "FreeLever")
    :param progname: Name of command-line tool running the function.
    :param filename: Filename of the input file (overwritten).
    :param N: Number of blocks in x-direction (the height of a plate == N /4).
    :param seed: Base seed to use to generate the disorder.
    :param k_drive: Stiffness of the drive string.
    :param symmetric: Set the drive string symmetric.
    :param test_mode: Run in test mode (smaller chunk).
    """

    assert not os.path.isfile(filename)

    M = max(3, int(N / 4))
    h = np.pi
    L = h * float(N)
    nlayer = 2 * nplates - 1

    is_plastic = np.zeros((nlayer), dtype=bool)
    is_plastic[1::2] = True

    drive = np.zeros((nlayer, 2), dtype=bool)
    for i, ispl in enumerate(is_plastic):
        if not ispl and i > 0:
            drive[i, 0] = True

    layer_bot = mesh.BottomLayerElastic(N, M, h)
    layer_top = mesh.TopLayerElastic(N, M, h)
    layer_elas = mesh.LayerElastic(N, M, h)
    layer_plas = GooseFEM.Mesh.Quad4.Regular(N, 1, h)

    stitch = GooseFEM.Mesh.Vstack()
    left = []
    right = []

    stitch.push_back(
        layer_bot.coor(),
        layer_bot.conn(),
        layer_bot.nodesBottomEdge(),
        layer_bot.nodesTopEdge(),
    )
    stitch.push_back(
        layer_plas.coor(),
        layer_plas.conn(),
        layer_plas.nodesBottomEdge(),
        layer_plas.nodesTopEdge(),
    )
    left += [layer_bot.nodesLeftOpenEdge(), layer_plas.nodesLeftEdge()]
    right += [layer_bot.nodesRightOpenEdge(), layer_plas.nodesRightEdge()]

    if nplates > 2:
        for i in range(nplates - 2):
            stitch.push_back(
                layer_elas.coor(),
                layer_elas.conn(),
                layer_elas.nodesBottomEdge(),
                layer_elas.nodesTopEdge(),
            )
            stitch.push_back(
                layer_plas.coor(),
                layer_plas.conn(),
                layer_plas.nodesBottomEdge(),
                layer_plas.nodesTopEdge(),
            )
            left += [layer_elas.nodesLeftOpenEdge(), layer_plas.nodesLeftEdge()]
            right += [layer_elas.nodesRightOpenEdge(), layer_plas.nodesRightEdge()]

    stitch.push_back(
        layer_top.coor(),
        layer_top.conn(),
        layer_top.nodesBottomEdge(),
        layer_top.nodesTopEdge(),
    )
    left += [layer_top.nodesLeftOpenEdge()]
    right += [layer_top.nodesRightOpenEdge()]

    left = stitch.nodeset(left)
    right = stitch.nodeset(right)
    bottom = stitch.nodeset(layer_bot.nodesBottomEdge(), 0)
    top = stitch.nodeset(layer_top.nodesTopEdge(), nlayer - 1)

    nelem = stitch.nelem()
    coor = stitch.coor()
    conn = stitch.conn()

    L = np.max(coor[:, 0]) - np.min(coor[:, 0])

    Hi = []
    for i in range(nlayer):
        yl = coor[conn[stitch.elemmap(i)[0], 0], 1]
        yu = coor[conn[stitch.elemmap(i)[-1], 3], 1]
        Hi += [0.5 * (yu + yl)]

    dofs = stitch.dofs()
    dofs[right, :] = dofs[left, :]
    dofs[top[-1], :] = dofs[top[0], :]
    dofs[bottom[-1], :] = dofs[bottom[0], :]
    dofs = GooseFEM.Mesh.renumber(dofs)

    iip = np.concatenate((dofs[bottom[:-1], :].ravel(), dofs[top[:-1], 1].ravel()))

    elastic = []
    plastic = []

    for i, ispl in enumerate(is_plastic):
        if ispl:
            plastic += list(stitch.elemmap(i))
        else:
            elastic += list(stitch.elemmap(i))

    initstate = seed + np.arange(N * (nplates - 1)).astype(np.int64)
    initseq = np.zeros_like(initstate)

    k = 2.0
    eps0 = 0.5 * 1e-4
    eps_offset = 1e-2 * (2.0 * eps0)
    nchunk = 6000

    if test_mode:
        nchunk = 200

    c = 1.0
    G = 1.0
    K = 4.5 * G  # consistent with PMMA
    rho = G / c ** 2.0
    qL = 2.0 * np.pi / L
    qh = 2.0 * np.pi / h
    alpha = np.sqrt(2.0) * qL * c * rho

    dt = (1.0 / (c * qh)) / 10.0

    with h5py.File(filename, "w") as file:

        storage.dump_with_atttrs(
            file,
            "/coor",
            coor,
            desc="Nodal coordinates [nnode, ndim]",
        )

        storage.dump_with_atttrs(
            file,
            "/conn",
            conn,
            desc="Connectivity (Quad4: nne = 4) [nelem, nne]",
        )

        storage.dump_with_atttrs(
            file,
            "/dofs",
            dofs,
            desc="DOFs per node, accounting for semi-periodicity [nnode, ndim]",
        )

        storage.dump_with_atttrs(
            file,
            "/iip",
            iip,
            desc="Prescribed DOFs [nnp]",
        )

        storage.dump_with_atttrs(
            file,
            "/run/event/deps",
            eps0 * 1e-4,
            desc="Strain kick to apply",
        )

        storage.dump_with_atttrs(
            file,
            "/run/dt",
            dt,
            desc="Time step",
        )

        storage.dump_with_atttrs(
            file,
            "/rho",
            rho * np.ones(nelem),
            desc="Mass density [nelem]",
        )

        storage.dump_with_atttrs(
            file,
            "/damping/alpha",
            alpha * np.ones(nelem),
            desc="Damping coefficient (density) [nelem]",
        )

        storage.dump_with_atttrs(
            file,
            "/elastic/elem",
            elastic,
            desc="Elastic elements [nelem - N]",
        )

        storage.dump_with_atttrs(
            file,
            "/elastic/K",
            K * np.ones(len(elastic)),
            desc="Bulk modulus for elements in '/elastic/elem' [nelem - N]",
        )

        storage.dump_with_atttrs(
            file,
            "/elastic/G",
            G * np.ones(len(elastic)),
            desc="Shear modulus for elements in '/elastic/elem' [nelem - N]",
        )

        storage.dump_with_atttrs(
            file,
            "/cusp/elem",
            plastic,
            desc="Plastic elements with cusp potential [nplastic]",
        )

        storage.dump_with_atttrs(
            file,
            "/cusp/K",
            K * np.ones(len(plastic)),
            desc="Bulk modulus for elements in '/cusp/elem' [nplastic]",
        )

        storage.dump_with_atttrs(
            file,
            "/cusp/G",
            G * np.ones(len(plastic)),
            desc="Shear modulus for elements in '/cusp/elem' [nplastic]",
        )

        storage.dump_with_atttrs(
            file,
            "/cusp/epsy/initstate",
            initstate,
            desc="State to initialise prrng.pcg32_array",
        )

        storage.dump_with_atttrs(
            file,
            "/cusp/epsy/initseq",
            initseq,
            desc="Sequence to initialise prrng.pcg32_array",
        )

        storage.dump_with_atttrs(
            file,
            "/cusp/epsy/k",
            k,
            desc="Shape factor of Weibull distribution",
        )

        storage.dump_with_atttrs(
            file,
            "/cusp/epsy/eps0",
            eps0,
            desc="Normalisation: epsy(i + 1) - epsy(i) = 2.0 * eps0 * random + eps_offset",
        )

        storage.dump_with_atttrs(
            file,
            "/cusp/epsy/eps_offset",
            eps_offset,
            desc="Offset, see eps0",
        )

        storage.dump_with_atttrs(
            file,
            "/cusp/epsy/nchunk",
            nchunk,
            desc="Chunk size",
        )

        storage.dump_with_atttrs(
            file,
            "/meta/normalisation/N",
            N,
            desc="Number of blocks along each plastic layer",
        )

        storage.dump_with_atttrs(
            file,
            "/meta/normalisation/l",
            h,
            desc="Elementary block size",
        )

        storage.dump_with_atttrs(
            file,
            "/meta/normalisation/rho",
            rho,
            desc="Elementary density",
        )

        storage.dump_with_atttrs(
            file,
            "/meta/normalisation/G",
            G,
            desc="Uniform shear modulus == 2 mu",
        )

        storage.dump_with_atttrs(
            file,
            "/meta/normalisation/K",
            K,
            desc="Uniform bulk modulus == kappa",
        )

        storage.dump_with_atttrs(
            file,
            "/meta/normalisation/eps",
            eps0,
            desc="Typical yield strain",
        )

        storage.dump_with_atttrs(
            file,
            "/meta/normalisation/sig",
            2.0 * G * eps0,
            desc="== 2 G eps0",
        )

        storage.dump_with_atttrs(
            file,
            "/meta/seed_base",
            seed,
            desc="Basic seed == 'unique' identifier",
        )

        meta = file.create_group(f"/meta/{config}/{progname}")
        meta.attrs["version"] = version

        elemmap = stitch.elemmap()
        nodemap = stitch.nodemap()

        storage.dump_with_atttrs(
            file,
            "/layers/stored",
            np.arange(len(elemmap)).astype(np.int64),
            desc="Layers in simulation",
        )

        for i in range(len(elemmap)):
            file[f"/layers/{i:d}/elemmap"] = elemmap[i]
            file[f"/layers/{i:d}/nodemap"] = nodemap[i]

        storage.dump_with_atttrs(
            file,
            "/layers/is_plastic",
            is_plastic,
            desc="Per layer: true is the layer is plastic",
        )

        storage.dump_with_atttrs(
            file,
            "/drive/k",
            k_drive,
            desc="Stiffness of the spring providing the drive",
        )

        storage.dump_with_atttrs(
            file,
            "/drive/symmetric",
            bool(symmetric),
            desc="If false, the driving spring buckles under tension.",
        )

        storage.dump_with_atttrs(
            file,
            "/drive/drive",
            drive,
            desc="Per layer: true when the layer's mean position is actuated",
        )

        storage.dump_with_atttrs(
            file,
            "/drive/height",
            Hi,
            desc="Height of the loading frame of each layer",
        )

        if config == "FreeLever":

            storage.dump_with_atttrs(
                file,
                "/drive/H",
                10.0 * np.diff(Hi)[0],
                desc="Height of the spring driving the lever",
            )

        desc = '(end of increment). One entry per item in "/stored".'
        storage.create_extendible(file, "/stored", np.uint64, desc="List of stored increments")
        storage.create_extendible(file, "/t", np.float64, desc=f"Time {desc}")
        storage.create_extendible(file, "/kick", bool, desc=f"Kick {desc}")
        storage.create_extendible(file, "/yield_element", bool, desc=f"Yield element {desc}")

        storage.dset_extend1d(file, "/stored", 0, 0)
        storage.dset_extend1d(file, "/t", 0, 0.0)
        storage.dset_extend1d(file, "/kick", 0, True)
        storage.dset_extend1d(file, "/yield_element", 0, False)

        file["/drive/ubar/0"] = np.zeros(drive.shape, dtype=np.float64)
        file["/drive/ubar"].attrs["desc"] = f"Loading frame position per layer {desc}"

        file["/disp/0"] = np.zeros_like(coor)
        file["/disp"].attrs["desc"] = f"Displacement {desc}"

        if config == "FreeLever":
            d = f"Target position of the lever {desc}"
            storage.create_extendible(file, "/drive/lever/target", np.float64, desc=d)

            d = f"Actual position of the lever {desc}"
            storage.create_extendible(file, "/drive/lever/position", np.float64, desc=d)

            storage.dset_extend1d(file, "/drive/lever/target", 0, 0.0)
            storage.dset_extend1d(file, "/drive/lever/position", 0, 0.0)

        assert np.min(np.diff(read_epsy(file), axis=1)) > file["/run/event/deps"][...]


def colors(name: str = "plates") -> list:
    """
    Return color-cycle.

    :param name: ``"plates"`` or ``"interfaces"`` or ``"layers"``
    :return: List of colors.
    """

    if name.lower() == "plates":
        return [cm.tue()[0], cm.tue()[4], cm.tue()[6], cm.tue()[8], cm.tue()[10]]

    if name.lower() == "interfaces":
        return [cm.tue()[1], cm.tue()[5], cm.tue()[7], cm.tue()[9]]

    if name.lower() == "layers":
        return [
            cm.tue()[0],
            cm.tue()[1],
            cm.tue()[4],
            cm.tue()[5],
            cm.tue()[6],
            cm.tue()[7],
            cm.tue()[8],
            cm.tue()[9],
            cm.tue()[10],
        ]

    raise OSError("Unknown name, choose from 'plates' or 'interfaces'")


def markers(name: str = "plates") -> list:
    """
    Return marker-cycle.

    :param name: ``"plates"`` or ``"interfaces"``
    :return: List of markers.
    """

    if name.lower() == "plates":
        return ["o", "^", ">", "v", "<"]

    if name.lower() == "interfaces":
        return ["^", ">", "v", "<"]

    raise OSError("Unknown name, choose from 'plates' or 'interfaces'")


def create_check_meta(
    file: h5py.File,
    path: str,
    ver: str,
    deps: str,
    dev: bool = False,
) -> h5py.Group:
    """
    Create or read/check meta data. This function asserts that:

    -   There are no uncommitted changes.
    -   There are no version changes.

    :param file: HDF5 archive.
    :param path: Path in ``file``.
    :param ver: Version string.
    :param deps: List of dependencies.
    :param dev: Allow uncommitted changes.
    :return: Group to meta-data.
    """

    assert dev or not tag.has_uncommitted(ver)
    assert dev or not tag.any_has_uncommitted(deps)

    if path not in file:
        meta = file.create_group(path)
        meta.attrs["version"] = ver
        meta.attrs["dependencies"] = deps
        return meta

    meta = file[path]
    assert dev or tag.greater_equal(ver, meta.attrs["version"])
    assert dev or tag.all_greater_equal(deps, meta.attrs["dependencies"])
    return meta


def _restore_inc(config: str, file: h5py.File, system, inc: int):
    """
    Restore an increment.

    :param config: Name of the configuration ("FixedLever", or "FreeLever")
    :param file: Open simulation HDF5 archive (read-only).
    :param system: The system,
    :param inc: Increment number.
    """

    system.setT(file["/t"][inc])
    system.setU(file[f"/disp/{inc:d}"][...])
    system.layerSetTargetUbar(file[f"/drive/ubar/{inc:d}"][...])

    if config == "FreeLever":
        # lever position is computed from equilibrium
        system.setLeverTarget(file["/drive/lever/target"][inc])


def _init_event_driven(config: str, file: h5py.File, system):
    """
    Initialise event-driven perturbation.

    :param config: Name of the configuration ("FixedLever", or "FreeLever")
    :param file: Open simulation HDF5 archive (read-only).
    :param system: The system,
    """

    eps0 = file["/meta/normalisation/eps"][...]

    if config == "FixedLever":
        ubar = system.layerTargetUbar_affineSimpleShear(eps0, file["/drive/height"][...])
        system.initEventDriven(ubar, file["/drive/drive"][...])
    else:
        system.initEventDriven(eps0, file["/drive/drive"][...])

    file["/run/event/delta_u"] = system.eventDriven_deltaU()
    file["/run/event/delta_ubar"] = system.eventDriven_deltaUbar()
    file["/run/event/active"] = system.eventDriven_targetActive()

    if config == "FreeLever":
        file["/run/event/delta_lever"] = system.eventDriven_deltaLeverPosition()


def _restore_event_driven(config: str, file: h5py.File, system):
    """
    Restore event driven settings.

    :param config: Name of the configuration ("FixedLever", or "FreeLever")
    :param file: Open simulation HDF5 archive (read-only).
    :param system: The system,
    """

    if config == "FixedLever":
        system.initEventDriven(
            file["/run/event/delta_ubar"][...],
            file["/run/event/active"][...],
            file["/run/event/delta_u"][...],
        )
    else:
        system.initEventDriven(
            file["/run/event/delta_lever"][...],
            file["/run/event/active"][...],
            file["/run/event/delta_u"][...],
            file["/run/event/delta_ubar"][...],
        )


def run(
    config: str,
    progname: str,
    model,
    init_function,
    filepath: str,
    dev: bool = False,
    progress: bool = True,
):
    """
    Run the simulation.

    :param config: Name of the configuration ("FixedLever", or "FreeLever")
    :param progname: Name of command-line tool running the function.
    :param model: Relevant FrictionQPotFEM module.
    :param init_function: Initialisation function.
    :param filepath: Name of the input/output file (appended).
    :param dev: Allow uncommitted changes.
    :param progress: Show progress bar.
    """

    basename = os.path.basename(filepath)

    with h5py.File(filepath, "a") as file:

        system = init_function(file)

        if config == "FreeLever":
            system.setLeverProperties(file["/drive/H"][...], file["/drive/height"][...])

        meta = f"/meta/{config}/{progname}"
        meta = create_check_meta(file, meta, version, dependencies(model), dev)

        if "completed" in meta:
            if progress:
                print(f'"{basename}": marked completed, skipping')
            return 1

        inc = int(file["/stored"][-1])
        deps = file["/run/event/deps"][...]
        kick = file["/kick"][inc]
        _restore_inc(config, file, system, inc)

        if "/run/event/delta_u" not in file:
            _init_event_driven(config, file, system)
        else:
            _restore_event_driven(config, file, system)

        nchunk = file["/cusp/epsy/nchunk"][...] - 5
        pbar = tqdm.tqdm(total=nchunk, disable=not progress)
        pbar.n = np.max(system.plastic_CurrentIndex())
        pbar.set_description(f"inc = {inc:8d}, loaded from file")
        pbar.refresh()

        for inc in range(inc + 1, sys.maxsize):

            yield_element = False
            kick = not kick

            if kick:
                idx_n = system.plastic_CurrentIndex()

            system.eventDrivenStep(deps, kick, direction=+1, yield_element=False, iterative=True)

            if kick:

                niter = system.minimise_boundcheck(5)
                idx = system.plastic_CurrentIndex()

                if niter == 0:
                    break

                if np.all(idx == idx_n):
                    _restore_inc(config, file, system, inc - 1)
                    system.eventDrivenStep(
                        deps, kick, direction=+1, yield_element=True, iterative=True
                    )
                    niter = system.minimise_boundcheck(5)
                    yield_element = True

                if niter == 0:
                    break

                if progress:
                    pbar.n = np.max(system.plastic_CurrentIndex())
                    pbar.set_description(f"inc = {inc:8d}, niter = {niter:8d}")
                    pbar.refresh()

            if not kick:
                if not system.boundcheck_right(5):
                    break

            storage.dset_extend1d(file, "/stored", inc, inc)
            storage.dset_extend1d(file, "/t", inc, system.t())
            storage.dset_extend1d(file, "/kick", inc, kick)
            storage.dset_extend1d(file, "/yield_element", inc, yield_element)
            file[f"/disp/{inc:d}"] = system.u()
            file[f"/drive/ubar/{inc:d}"] = system.layerTargetUbar()

            if config == "FreeLever":
                storage.dset_extend1d(file, "/drive/lever/target", inc, system.leverTarget())
                storage.dset_extend1d(file, "/drive/lever/position", inc, system.leverPosition())

            file.flush()

        if progress:
            pbar.n = np.max(nchunk)
            pbar.set_description("{:32s}".format("completed"))
            pbar.refresh()

        meta.attrs["completed"] = 1


def runinc_event_basic(config: str, system, file: h5py.File, inc: int, Smax=None) -> dict:
    """
    Rerun increment and get basic event information.

    :param config: Name of the configuration ("FixedLever", or "FreeLever")
    :param system: The system (modified: increment loaded/rerun).
    :param file: Open simulation HDF5 archive (read-only).
    :param inc: The increment to rerun.
    :param Smax: Stop at given S (to avoid spending time on final energy minimisation).
    :return: A dictionary as follows::

        r: Position of yielding event (block index).
        t: Time of each yielding event.
        S: Size (signed) of the yielding event.
    """

    stored = file["/stored"][...]

    if Smax is None:
        Smax = np.inf

    assert inc > 0
    assert inc in stored
    assert inc - 1 in stored

    _restore_inc(config, file, system, inc - 1)
    _restore_event_driven(config, file, system)

    idx_n = system.plastic_CurrentIndex()[:, 0].astype(int)
    idx_t = system.plastic_CurrentIndex()[:, 0].astype(int)

    system.eventDrivenStep(
        file["/run/event/deps"][...],
        file["/kick"][inc],
        direction=+1,
        yield_element=file["/yield_element"][inc],
        iterative=True,
    )

    R = []
    T = []
    S = []

    while True:

        niter = system.timeStepsUntilEvent()

        idx = system.plastic_CurrentIndex()[:, 0].astype(int)
        t = system.t()

        for r in np.argwhere(idx != idx_t):
            R += [r]
            T += [t * np.ones(r.shape)]
            S += [(idx - idx_t)[r]]

        idx_t = np.array(idx, copy=True)

        if np.sum(idx - idx_n) >= Smax:
            break

        if niter == 0:
            break

    ret = dict(r=np.array(R).ravel(), t=np.array(T).ravel(), S=np.array(S).ravel())

    # check docstring
    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    d = doc.split(":return:")[1].split("\n", 2)[2]
    _check_output(yaml.safe_load(d), ret)

    return ret


def _check_output(a: Union[dict, list], b: Union[dict, list]):
    """
    Check if a dictionary/list ``a`` has all fields as ``b`` and vice-versa.

    :param a: List or dictionary.
    :param b: List or dictionary.
    """

    for key in a:
        if key not in b:
            raise OSError(f"{key} not in b")

    for key in b:
        if key not in a:
            raise OSError(f"{key} not in a")


def basic_output(
    system,
    file: h5py.File,
    verbose: bool = True,
) -> dict:
    """
    Read basic output from simulation.

    :param system: The system (modified: all increments visited).
    :param file: Open simulation HDF5 archive (read-only).
    :param verbose: Print progress.
    :return: A dictionary as follows::

        epsd: np.array([ninc]) # macroscopic strain, units: eps0
        sigd: np.array([ninc]) # macroscopic stress, units: sig0
        epsd_layers: np.array([ninc, nlayer]) # strain averaged per layer, units: eps0
        sigd_layers: np.array([ninc, nlayer]) # stress averaged per layer, units: sig0
        S: np.array([ninc]) # number of times that blocks yielded
        A: np.array([ninc]) # number blocks that yielded
        S_layers: np.array([ninc, nlayer]) # S per layer
        A_layers: np.array([ninc, nlayer]) # A per layer
        ubarx_layers: np.array([ninc, nlayer]) # average x-displacement per layer
        fxdrive_layers: np.array([ninc, nlayer]) # reaction for per layer: only valid if drive[:, 0]
        gamma: np.array([ninc]) # rotation of the lever, units: degrees
        inc: np.array([ninc]) # increment numbers
        kick: np.array([ninc]) # kick (True) or elastic load (False)
        steadystate: int # first steadystate increment (or infinity)
        N: int # number of blocks in horizontal direction
        nlayer: int # number of layers
        is_plastic: np.array([nlayer]) # layer set plastic (True) or elastic (False)
        drive: np.array([nlayer]) # layer is driven (True) or free (False)
        height: np.array([nlayer]) # height of the driving spring of each layer
        volume: float # volume of the system
        eps0: float # typical yield strain
        sig0: float # typical yield stress
        G: float # shear modulus
        K: float # bulk modulus
        rho: float # mass density
        cs: float # shear wave speed
        cd: float # dilatational wave speed
        l0: float # linear block size
        t0: float # typical time == l0 / cs
        dt: float # time step
        kdrive: float # stiffness of the driving spring
        seed: int # base seed
    """

    incs = file["/stored"][...]
    ninc = incs.size
    assert np.all(incs == np.arange(ninc))
    nlayer = system.nlayer()

    ret = dict(
        epsd=np.zeros((ninc), dtype=float),
        sigd=np.zeros((ninc), dtype=float),
        epsd_layers=np.zeros((ninc, nlayer), dtype=float),
        sigd_layers=np.zeros((ninc, nlayer), dtype=float),
        S_layers=np.zeros((ninc, nlayer), dtype=int),
        A_layers=np.zeros((ninc, nlayer), dtype=int),
        ubarx_layers=np.zeros((ninc, nlayer), dtype=float),
        fxdrive_layers=np.zeros((ninc, nlayer), dtype=float),
        S=np.zeros((ninc), dtype=int),
        A=np.zeros((ninc), dtype=int),
        gamma=np.zeros((ninc), dtype=float),
        inc=incs,
        kick=file["/kick"][incs],
        N=file["/meta/normalisation/N"][...],
        nlayer=nlayer,
        is_plastic=file["/layers/is_plastic"][...],
        drive=file["/drive/drive"][...],
        height=file["/drive/height"][...],
        eps0=file["/meta/normalisation/eps"][...],
        sig0=file["/meta/normalisation/sig"][...],
        G=file["/meta/normalisation/G"][...],
        K=file["/meta/normalisation/K"][...],
        rho=file["/meta/normalisation/rho"][...],
        l0=file["/meta/normalisation/l"][...],
        dt=file["/run/dt"][...],
        kdrive=file["/drive/k"][...],
        seed=file["/meta/seed_base"][...],
    )

    kappa = ret["K"] / 3.0
    mu = ret["G"] / 2.0
    ret["cs"] = np.sqrt(mu / ret["rho"])
    ret["cd"] = np.sqrt((kappa + 4.0 / 3.0 * mu) / ret["rho"])
    ret["t0"] = ret["l0"] / ret["cs"]

    dV = system.quad().dV()
    ret["volume"] = np.sum(dV)
    dV = system.quad().AsTensor(2, dV)
    idx_n = system.plastic_CurrentIndex().astype(int)[:, 0].reshape(-1, ret["N"])
    idx_0 = np.array(idx_n, copy=True)
    ret["steadystate"] = None

    for inc in tqdm.tqdm(incs, disable=not verbose):

        ubar = file[f"/drive/ubar/{inc:d}"][...]
        system.layerSetTargetUbar(ubar)

        u = file[f"/disp/{inc:d}"][...]
        system.setU(u)

        ret["gamma"][inc] = 90 - np.rad2deg(np.arctan2(ret["height"][-1], ubar[-1, 0]))
        Sig = system.Sig() / ret["sig0"]
        Eps = system.Eps() / ret["eps0"]
        idx = system.plastic_CurrentIndex().astype(int)[:, 0].reshape(-1, ret["N"])

        for i in range(nlayer):
            e = system.layerElements(i)
            E = np.average(Eps[e, ...], weights=dV[e, ...], axis=(0, 1))
            S = np.average(Sig[e, ...], weights=dV[e, ...], axis=(0, 1))
            ret["epsd_layers"][inc, i] = GMat.Epsd(E)
            ret["sigd_layers"][inc, i] = GMat.Sigd(S)

        ret["epsd"][inc] = GMat.Epsd(np.average(Eps, weights=dV, axis=(0, 1)))
        ret["sigd"][inc] = GMat.Sigd(np.average(Sig, weights=dV, axis=(0, 1)))
        ret["S"][inc] = np.sum(idx - idx_n)
        ret["A"][inc] = np.sum(idx != idx_n)
        ret["S_layers"][inc, ret["is_plastic"]] = np.sum(idx - idx_n, axis=1)
        ret["A_layers"][inc, ret["is_plastic"]] = np.sum(idx != idx_n, axis=1)
        ret["fxdrive_layers"][inc, :] = system.layerFdrive()[:, 0] / ret["kdrive"]
        ret["ubarx_layers"][inc, :] = system.layerUbar()[:, 0]

        if ret["steadystate"] is None:
            if np.all(idx != idx_0):
                ret["steadystate"] = inc

        idx_n = np.array(idx, copy=True)

    if ret["steadystate"] is None:
        ret["steadystate"] = sys.maxsize
    else:
        ret["steadystate"] = max(ret["steadystate"], steadystate(**ret))

    # check docstring
    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    d = doc.split(":return:")[1].split("\n", 2)[2]
    _check_output(yaml.safe_load(d), ret)

    return ret


def cli_generate(cli_args: list[str], entry_points: dict, config: str):
    """
    Generate IO files, including job-scripts to run simulations.
    """

    class MyFmt(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=doc)

    parser.add_argument("--develop", action="store_true", help="Allow uncommitted changes")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite")
    parser.add_argument("-n", "--nsim", type=int, default=1, help="#simulations")
    parser.add_argument("-N", "--size", type=int, default=2 * (3 ** 6), help="#blocks")
    parser.add_argument("-s", "--start", type=int, default=0, help="Start simulation")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("-w", "--time", type=str, default="72h", help="Walltime")
    parser.add_argument("outdir", type=str, help="Output directory")

    parser.add_argument(
        "-l",
        "--nlayer",
        type=int,
        default=6,
        help="Maximum number of layers (generated: 2, 3, ..., nlayer)",
    )

    parser.add_argument(
        "--max-plates",
        type=int,
        default=100,
        help="Maximum number of plates, help to set the seeds.",
    )

    parser.add_argument(
        "-k",
        type=float,
        default=1e-3,
        help="Stiffness of the drive spring (typically low)",
    )

    parser.add_argument(
        "--symmetric",
        type=int,
        default=1,
        help="Set the symmetry of the drive spring (True/False)",
    )

    if cli_args is None:
        args = parser.parse_args(sys.argv[1:])
    else:
        args = parser.parse_args([str(arg) for arg in cli_args])

    assert args.develop or not tag.has_uncommitted(version)
    assert os.path.isdir(args.outdir)

    files = []

    for i, nplates in itertools.product(
        range(args.start, args.start + args.nsim), range(2, args.nlayer + 1)
    ):
        filename = "_".join(
            [
                f"id={i:03d}",
                f"nplates={nplates:d}",
                f"kplate={args.k:.0e}",
                f"symmetric={args.symmetric:d}.h5",
            ]
        )
        files += [filename]
        filepath = os.path.join(args.outdir, filename)

        if args.force and os.path.isfile(filepath):
            os.remove(filepath)

        generate(
            config=config,
            progname=entry_points["cli_generate"],
            filename=filepath,
            N=args.size,
            nplates=nplates,
            seed=i * args.size * (args.max_plates - 1),
            k_drive=args.k,
            symmetric=args.symmetric,
        )

    progname = entry_points["cli_run"]
    commands = [f"{progname} {file}" for file in files]
    slurm.serial_group(
        commands,
        basename=progname,
        group=1,
        outdir=args.outdir,
        sbatch={"time": args.time},
    )


def cli_run(cli_args: list[str], entry_points: dict, config: str, model, init_function):
    """
    Run simulation.
    """

    class MyFmt(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=doc)

    parser.add_argument("--develop", action="store_true", help="Allow uncommitted changes")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("file", type=str, help="Simulation file")

    if cli_args is None:
        args = parser.parse_args(sys.argv[1:])
    else:
        args = parser.parse_args([str(arg) for arg in cli_args])

    assert os.path.isfile(args.file)

    run(
        config=config,
        progname=entry_points[funcname],
        model=model,
        init_function=init_function,
        filepath=args.file,
        dev=args.develop,
    )


def cli_copy_perturbation(cli_args: list[str], entry_points: dict, config: str):
    """
    Copy pre-computed perturbation to be used in then event driven protocol to
    (a bunch of) un-run simulations.
    """

    class MyFmt(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=doc)

    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("source", type=str, help="Source")
    parser.add_argument("files", nargs="*", type=str, help="Destination(s)")

    if cli_args is None:
        args = parser.parse_args(sys.argv[1:])
    else:
        args = parser.parse_args([str(arg) for arg in cli_args])

    assert [os.path.isfile(args.source)]
    assert [os.path.isfile(f) for f in args.files]

    with h5py.File(args.source, "r") as file:

        coor = file["coor"][...]
        conn = file["conn"][...]
        dofs = file["dofs"][...]
        iip = file["iip"][...]
        active = file["/run/event/active"][...]
        delta_u = file["/run/event/delta_u"][...]
        delta_ubar = file["/run/event/delta_ubar"][...]

        if config == "FreeLever":
            delta_lever = file["/run/event/delta_lever"][...]

    for filepath in args.files:

        with h5py.File(filepath, "a") as file:

            assert "/run/event/delta_ubar" not in file
            assert "/run/event/active" not in file
            assert "/run/event/delta_u" not in file
            assert config == "FixedLever" or "/run/event/delta_ubar" not in file
            assert config == "FixedLever" or "/drive/H" in file
            assert config == "FreeLever" or "/drive/H" not in file
            assert np.allclose(coor, file["coor"][...])
            assert np.all(np.equal(conn, file["conn"][...]))
            assert np.all(np.equal(dofs, file["dofs"][...]))
            assert np.all(np.equal(iip, file["iip"][...]))

            file["/run/event/active"] = active
            file["/run/event/delta_u"] = delta_u
            file["/run/event/delta_ubar"] = delta_ubar

            if config == "FreeLever":
                file["/run/event/delta_lever"] = delta_lever


def cli_rerun_event(
    cli_args: list[str], entry_points: dict, file_defaults: dict, config: str, model
):
    """
    Rerun increments and store basic event info (position and time).
    Tip: truncate when (know) S is reached to not waste time on final stage of energy minimisation.
    """

    class MyFmt(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=replace_entry_point(doc))
    progname = entry_points[funcname]
    output = file_defaults[funcname]

    parser.add_argument("-s", "--smax", type=int, help="Truncate at a maximum total S")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite")
    parser.add_argument("-i", "--inc", required=True, type=int, help="Increment number")
    parser.add_argument("-o", "--output", type=str, default=output, help="Output file")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("file", type=str, help="Simulation file")

    if cli_args is None:
        args = parser.parse_args(sys.argv[1:])
    else:
        args = parser.parse_args([str(arg) for arg in cli_args])

    assert os.path.isfile(args.file)

    if not args.force:
        if os.path.isfile(args.output):
            if not click.confirm(f'Overwrite "{args.output}"?'):
                raise OSError("Cancelled")

    with h5py.File(args.file, "r") as file:
        system = init(file, model)
        ret = runinc_event_basic(config, system, file, args.inc, args.smax)

    with h5py.File(args.output, "w") as file:
        file["r"] = ret["r"]
        file["t"] = ret["t"]
        file["S"] = ret["S"]
        meta = file.create_group(f"/meta/{config}/{progname}")
        meta.attrs["version"] = version
        meta.attrs["dependencies"] = dependencies(model)


def cli_job_rerun_multislip(cli_args: list[str], entry_points: dict):
    """
    Rerun increments that have events in which more than one layer slips.
    To consider a layer as having slipped an arbitrary fraction of blocks need to have slipped
    on that layer.
    """

    class MyFmt(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=replace_entry_point(doc))
    executable = entry_points["cli_rerun_event"]

    parser.add_argument("-b", "--basename", type=str, default=executable, help="Output base")
    parser.add_argument("-m", "--min", type=float, default=0.5, help="fraction of slipping blocks")
    parser.add_argument("-n", "--group", type=int, default=20, help="#pushes to group per job")
    parser.add_argument("-o", "--outdir", type=str, default=os.getcwd(), help="Output directory")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("-w", "--time", type=str, default="24h", help="Walltime")
    parser.add_argument("info", type=str, help="EnsembleInfo (read-only)")

    if cli_args is None:
        args = parser.parse_args(sys.argv[1:])
    else:
        args = parser.parse_args([str(arg) for arg in cli_args])

    assert os.path.isfile(args.info)

    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    basedir = os.path.dirname(args.info)

    commands = []

    with h5py.File(args.info, "r") as file:

        N = file["/normalisation/N"][...]

        for full in file["/full"].attrs["stored"]:
            S = file[f"/full/{full}/S_layers"][...]
            A = file[f"/full/{full}/A_layers"][...]
            ss = file[f"/full/{full}/steadystate"][...]
            nany = np.sum(S > 0, axis=1)
            nall = np.sum(A >= args.min * N, axis=1)
            incs = np.argwhere(np.logical_and(nany > 1, nall >= 1)).ravel()
            incs = incs[incs > ss]

            if len(incs) == 0:
                continue

            simid = os.path.splitext(os.path.basename(full))[0]
            filepath = os.path.join(basedir, full)
            relfile = os.path.relpath(filepath, args.outdir)

            for i in incs:
                s = np.sum(S[i, :])
                commands += [f"{executable} -i {i:d} -s {s:d} -o {simid}_inc={i:d}.h5 {relfile}"]

    slurm.serial_group(
        commands,
        basename=args.basename,
        group=args.group,
        outdir=args.outdir,
        sbatch={"time": args.time},
    )


def cli_plot(cli_args: list[str], init_function):
    """
    Plot basic output.
    """

    class MyFmt(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=replace_entry_point(doc))

    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite")
    parser.add_argument("-m", "--marker", type=str, help="Set marker")
    parser.add_argument("-o", "--output", type=str, help="Output file")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("file", type=str, help="File to read")

    if cli_args is None:
        args = parser.parse_args(sys.argv[1:])
    else:
        args = parser.parse_args([str(arg) for arg in cli_args])

    assert os.path.isfile(args.file)

    if args.output:
        if not args.force:
            if os.path.isfile(args.output):
                if not click.confirm(f'Overwrite "{args.output}"?'):
                    raise OSError("Cancelled")

    with h5py.File(args.file, "r") as file:
        system = init_function(file)
        out = basic_output(system, file, verbose=False)

    opts = {}
    if args.marker:
        opts["marker"] = args.marker

    fig, axes = gplt.subplots(ncols=2)
    axes[0].plot(out["epsd"], out["sigd"], **opts)
    axes[1].plot(out["epsd"], np.cumsum(out["S_layers"], axis=0), **opts)

    if args.output:
        fig.savefig(args.output)
    else:
        plt.show()

    plt.close(fig)
