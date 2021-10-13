import argparse
import inspect
import itertools
import os
import re
import sys
import textwrap

import click
import FrictionQPotFEM.UniformMultiLayerIndividualDrive2d as model
import GMatElastoPlasticQPot.Cartesian2d as GMat
import GooseFEM
import GooseHDF5 as g5
import h5py
import numpy as np
import prrng
import tqdm
import XDMFWrite_h5py as xh
import matplotlib.pyplot as plt
import GooseMPL as gplt

from collections import defaultdict

plt.style.use(["goose", "goose-latex"])

from . import mesh
from . import slurm
from . import storage
from . import System
from . import tag
from ._version import version

config = "FixedLever"

entry_points = dict(
    cli_run="FixedLever_Run",
    cli_generate="FixedLever_Generate",
    cli_compare="FixedLever_Compare",
    cli_plot="FixedLever_Plot",
    cli_ensembleinfo="FixedLever_EnsembleInfo",
    cli_view_paraview="FixedLever_Paraview",
    cli_rerun_event="FixedLever_Events",
    cli_job_rerun_multislip="FixedLever_EventsJob",
)

file_defaults = dict(
    cli_ensembleinfo="EnsembleInfo.h5",
    cli_rerun_event="FixedLever_Events.h5",
)


def dependencies(system: model.System) -> list[str]:
    """
    Return list with version strings.
    Compared to model.System.version_dependencies() this added the version of prrng.
    """
    return sorted(list(model.version_dependencies()) + ["prrng=" + prrng.version()])


def replace_entry_point(docstring):
    """
    Replace ":py:func:`...`" with the relevant entry_point name
    """
    for ep in entry_points:
        docstring = docstring.replace(fr":py:func:`{ep:s}`", entry_points[ep])
    return docstring


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


def generate(
    filename: str,
    N: int,
    nplates: int,
    seed: int,
    k_drive: float,
    symmetric: bool = True,
    delta_gamma: float = None,
):
    """
    Generate an input file.

    :param filename: Filename of the input file (overwritten).
    :param N: Number of blocks in x-direction (the height of a plate == N /4).
    :param seed: Base seed to use to generate the disorder.
    :param k_drive: Stiffness of the drive string.
    :param symmetric: Set the drive string symmetric.
    :param delta_gamma: Loading history (for testing only).
    """

    assert not os.path.isfile(os.path.realpath(filename))

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
    # H = np.max(coor[:, 1]) - np.min(coor[:, 1])

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

    # generators = prrng.pcg32_array(initstate, initseq)
    # epsy = eps_offset + (2.0 * eps0) * generators.weibull([nchunk], k)
    # epsy[0: left, 0] *= init_factor
    # epsy[right: N, 0] *= init_factor
    # epsy = np.cumsum(epsy, axis=1)

    if delta_gamma is None:
        delta_gamma = 0.001 * eps0 * np.ones(10000) / k_drive
        delta_gamma[0] = 0

    c = 1.0
    G = 1.0
    K = 4.5 * G  # consistent with PMMA
    rho = G / c ** 2.0
    qL = 2.0 * np.pi / L
    qh = 2.0 * np.pi / h
    alpha = np.sqrt(2.0) * qL * c * rho

    dt = (1.0 / (c * qh)) / 10.0

    progname = entry_points["cli_generate"]

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
            "/run/epsd/kick",
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
            "/drive/delta_gamma",
            delta_gamma,
            desc="Affine simple shear increment",
        )

        storage.dump_with_atttrs(
            file,
            "/drive/height",
            Hi,
            desc="Height of the loading frame of each layer",
        )


def cli_generate(cli_args=None):
    """
    Generate IO files, including job-scripts to run simulations.
    """

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    docstring = textwrap.dedent(inspect.getdoc(globals()[funcname]))

    if cli_args is None:
        cli_args = sys.argv[1:]
    else:
        cli_args = [str(arg) for arg in cli_args]

    class MyFormatter(
        argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
    ):
        pass

    parser = argparse.ArgumentParser(
        formatter_class=MyFormatter, description=replace_entry_point(docstring)
    )

    parser.add_argument("outdir", type=str, help="Output directory")
    parser.add_argument("-N", "--size", type=int, default=2 * (3 ** 6), help="#blocks")
    parser.add_argument("-n", "--nsim", type=int, default=1, help="#simulations")
    parser.add_argument("-s", "--start", type=int, default=0, help="Start simulation")
    parser.add_argument("-w", "--time", type=str, default="72h", help="Walltime")
    parser.add_argument("-v", "--version", action="version", version=version)

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

    args = parser.parse_args(cli_args)

    assert os.path.isdir(os.path.realpath(args.outdir))

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

        generate(
            filename=os.path.join(args.outdir, filename),
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

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    docstring = textwrap.dedent(inspect.getdoc(globals()[funcname]))

    if cli_args is None:
        cli_args = sys.argv[1:]
    else:
        cli_args = [str(arg) for arg in cli_args]

    class MyFormatter(
        argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
    ):
        pass

    parser = argparse.ArgumentParser(
        formatter_class=MyFormatter, description=replace_entry_point(docstring)
    )

    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("--develop", action="store_true", help="For testing")
    parser.add_argument("file_a", type=str, help="Simulation file")
    parser.add_argument("file_b", type=str, help="Simulation file")

    args = parser.parse_args(cli_args)

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

    if args.develop:
        return ret
    else:
        return 0


def run(filename: str, dev: bool):
    """
    Run the simulation.

    :param filename: Name of the input/output file (appended).
    :param dev: Allow uncommitted changes.
    """

    basename = os.path.basename(filename)
    progname = entry_points["cli_run"]

    with h5py.File(filename, "a") as file:

        system = System.init(file)

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

        assert tag.greater_equal(version, meta.attrs["version"])
        assert tag.all_greater_equal(dependencies(model), meta.attrs["dependencies"])

        # restore or initialise the system / output

        if "/stored" in file:

            inc = int(file["/stored"][-1])
            system.setT(file["/t"][inc])
            system.setU(file[f"/disp/{inc:d}"][...])
            system.layerSetTargetUbar(file[f"/drive/ubar/{inc:d}"][...])
            print(f'"{basename}": Loading, inc = {inc:d}')

        else:

            inc = int(0)

            storage.dset_extendible1d(
                file=file,
                key="/stored",
                dtype=np.uint64,
                value=inc,
                desc="List of stored increments",
            )

            storage.dset_extendible1d(
                file=file,
                key="/t",
                dtype=np.float64,
                value=system.t(),
                desc="Per increment: time at the end of the increment",
            )

            storage.dump_with_atttrs(
                file=file,
                key=f"/disp/{inc}",
                data=system.u(),
                desc="Displacement (end of increment).",
            )

            storage.dump_with_atttrs(
                file=file,
                key=f"/drive/ubar/{inc}",
                data=system.layerTargetUbar(),
                desc="Loading frame position per layer.",
            )

        # run

        height = file["/drive/height"][...]
        delta_gamma = file["/drive/delta_gamma"][...]

        assert np.isclose(delta_gamma[0], 0.0)
        inc += 1

        for dgamma in delta_gamma[inc:]:

            system.layerTagetUbar_addAffineSimpleShear(dgamma, height)
            niter = system.minimise()
            if not system.boundcheck_right(5):
                break
            print(f'"{basename}": inc = {inc:8d}, niter = {niter:8d}')

            storage.dset_extend1d(file, "/stored", inc, inc)
            storage.dset_extend1d(file, "/t", inc, system.t())
            file[f"/disp/{inc:d}"] = system.u()
            file[f"/drive/ubar/{inc:d}"] = system.layerTargetUbar()

            inc += 1

        meta.attrs["completed"] = 1


def cli_run(cli_args=None):
    """
    Run simulation.
    """

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    docstring = textwrap.dedent(inspect.getdoc(globals()[funcname]))

    if cli_args is None:
        cli_args = sys.argv[1:]
    else:
        cli_args = [str(arg) for arg in cli_args]

    class MyFormatter(
        argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
    ):
        pass

    parser = argparse.ArgumentParser(
        formatter_class=MyFormatter, description=replace_entry_point(docstring)
    )

    parser.add_argument("-f", "--force", action="store_true", help="Allow uncommitted")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("file", type=str, help="Simulation file")

    args = parser.parse_args(cli_args)

    assert os.path.isfile(os.path.realpath(args.file))
    run(args.file, dev=args.force)


def runinc_event_basic(
    system: model.System, file: h5py.File, inc: int, Smax=sys.maxsize
) -> dict:
    """
    Rerun increment and get basic event information.

    :param system: The system (modified: all increments visited).
    :param file: Open simulation HDF5 archive (read-only).
    :param inc: The increment to return.
    :param Smax: Optionally truncate the run at a given total S.

    :return: Dictionary:
        ``r`` : Position with columns (layer, block).
        ``t`` : Time of each row in ``r``.
    """

    stored = file["/stored"][...]
    N = file["/meta/normalisation/N"][...]

    assert inc > 0
    assert inc in stored
    assert inc - 1 in stored

    system.layerSetTargetUbar(file[f"/drive/ubar/{inc - 1:d}"][...])
    system.setU(file[f"/disp/{inc - 1:d}"][...])

    idx_n = system.plastic_CurrentIndex().astype(int)[:, 0].reshape(-1, N)
    idx_t = system.plastic_CurrentIndex().astype(int)[:, 0].reshape(-1, N)

    height = file["/drive/height"][...]
    dgamma = file["/drive/delta_gamma"][inc]

    system.layerTagetUbar_addAffineSimpleShear(dgamma, height)

    R = []
    T = []

    while True:

        niter = system.timeStepsUntilEvent()

        idx = system.plastic_CurrentIndex().astype(int)[:, 0].reshape(-1, N)
        t = system.t()

        for r in np.argwhere(idx != idx_t):
            R += [list(r)]
            T += [t]

        idx_t = np.array(idx, copy=True)

        if np.sum(idx - idx_n) >= Smax:
            break

        if niter == 0:
            break

    return dict(r=np.array(R), t=np.array(T))


def cli_rerun_event(cli_args=None):
    """
    Rerun increments and store basic event info.
    """

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    docstring = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    progname = entry_points[funcname]
    output = file_defaults[funcname]

    if cli_args is None:
        cli_args = sys.argv[1:]
    else:
        cli_args = [str(arg) for arg in cli_args]

    class MyFormatter(
        argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
    ):
        pass

    parser = argparse.ArgumentParser(
        formatter_class=MyFormatter, description=replace_entry_point(docstring)
    )

    parser.add_argument(
        "-s",
        "--smax",
        type=int,
        help="Truncate at a given maximal total S",
    )

    parser.add_argument("-o", "--output", type=str, default=output, help="Output file")
    parser.add_argument("-i", "--inc", required=True, type=int, help="Increment number")
    parser.add_argument("-f", "--force", action="store_true", help="Allow uncommitted")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("file", type=str, help="Simulation file")

    args = parser.parse_args(cli_args)

    assert os.path.isfile(os.path.realpath(args.file))

    if not args.force:
        if os.path.isfile(args.output):
            if not click.confirm(f'Overwrite "{args.output}"?'):
                raise OSError("Cancelled")

    with h5py.File(args.file, "r") as file:
        system = System.init(file)
        if args.smax is None:
            ret = runinc_event_basic(system, file, args.inc)
        else:
            ret = runinc_event_basic(system, file, args.inc, args.smax)

    with h5py.File(args.output, "w") as file:
        file["r"] = ret["r"]
        file["t"] = ret["t"]
        meta = file.create_group(f"/meta/{config}/{progname}")
        meta.attrs["version"] = version
        meta.attrs["dependencies"] = dependencies(model)


def cli_job_rerun_multislip(cli_args=None):
    """
    Rerun increments that have events in which more than one layer slips.
    """

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    docstring = textwrap.dedent(inspect.getdoc(globals()[funcname]))

    if cli_args is None:
        cli_args = sys.argv[1:]
    else:
        cli_args = [str(arg) for arg in cli_args]

    class MyFormatter(
        argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
    ):
        pass

    parser = argparse.ArgumentParser(
        formatter_class=MyFormatter, description=replace_entry_point(docstring)
    )

    parser.add_argument("info", type=str, help="EnsembleInfo (read-only)")

    parser.add_argument(
        "-m",
        "--min",
        type=float,
        default=0.5,
        help="Minimum fraction of blocks that slips on one of the layers to select the event",
    )

    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        default=os.getcwd(),
        help="Output directory of all simulation and output variables",
    )

    parser.add_argument(
        "-c",
        "--conda",
        type=str,
        default=slurm.default_condabase,
        help="Base name of the conda environment, appended '_E5v4' and '_s6g1'",
    )

    parser.add_argument(
        "-n",
        "--group",
        type=int,
        default=50,
        help="Number of pushes to group in a single job",
    )

    parser.add_argument(
        "-w",
        "--time",
        type=str,
        default="24h",
        help="Wall-time to allocate for the job",
    )

    parser.add_argument(
        "-e",
        "--executable",
        type=str,
        default=entry_points["cli_rerun_event"],
        help="Executable to use",
    )

    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=version,
    )

    args = parser.parse_args(cli_args)

    assert os.path.isfile(os.path.realpath(args.info))
    assert os.path.isdir(os.path.realpath(args.outdir))

    basedir = os.path.dirname(args.info)
    executable = args.executable

    commands = []

    with h5py.File(args.info, "r") as file:

        N = file["/normalisation/N"][...]

        for full in file["/full"].attrs["stored"]:
            S = file[f"/full/{full}/S_layers"][...]
            A = file[f"/full/{full}/A_layers"][...]
            nany = np.sum(S > 0, axis=1)
            nall = np.sum(A >= args.min * N, axis=1)
            incs = np.argwhere((nany > 1) * (nall >= 1)).ravel()

            if len(incs) == 0:
                continue

            simid = os.path.splitext(os.path.basename(full))[0]
            filepath = os.path.join(basedir, full)
            relfile = os.path.relpath(filepath, args.outdir)

            for i in incs:
                s = np.sum(S[i, :])
                commands += [
                    f"{executable} -i {i:d} -s {s:d} -o {simid}_inc={i:d}.h5 {relfile}"
                ]

    slurm.serial_group(
        commands,
        basename=args.executable.replace(" ", "_"),
        group=args.group,
        outdir=args.outdir,
        sbatch={"time": args.time},
    )


def basic_output(
    system: model.System,
    file: h5py.File,
    verbose: bool = True,
) -> dict:
    """
    Read basic output from simulation.

    :param system: The system (modified: all increments visited).
    :param file: Open simulation HDF5 archive (read-only).
    :param verbose: Print progress.
    """

    incs = file["/stored"][...]
    ninc = incs.size
    assert np.all(incs == np.arange(ninc))
    nlayer = system.nlayer()
    progname = entry_points["cli_run"]

    ret = dict(
        epsd=np.empty((ninc), dtype=float),
        sigd=np.empty((ninc), dtype=float),
        epsd_layers=np.empty((ninc, nlayer), dtype=float),
        sigd_layers=np.empty((ninc, nlayer), dtype=float),
        S_layers=np.zeros((ninc, nlayer), dtype=int),
        A_layers=np.zeros((ninc, nlayer), dtype=int),
        inc=incs,
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
        version=file[f"/meta/{config}/{progname}"].attrs["version"],
        dependencies=file[f"/meta/{config}/{progname}"].attrs["dependencies"],
    )

    kappa = ret["K"] / 3.0
    mu = ret["G"] / 2.0
    ret["cs"] = np.sqrt(mu / ret["rho"])
    ret["cd"] = np.sqrt((kappa + 4.0 / 3.0 * mu) / ret["rho"])
    ret["t0"] = ret["l0"] / ret["cs"]

    ret["delta_gamma"] = file["/drive/delta_gamma"][...][incs] / ret["eps0"]

    dV = system.quad().AsTensor(2, system.quad().dV())
    idx_n = system.plastic_CurrentIndex().astype(int)[:, 0].reshape(-1, ret["N"])
    drive_x = np.argwhere(ret["drive"][:, 0]).ravel()

    ret["drive_ux"] = np.zeros((ninc, drive_x.size), dtype=float)
    ret["drive_fx"] = np.zeros((ninc, drive_x.size), dtype=float)
    ret["layers_ux"] = np.zeros((ninc, nlayer), dtype=float)
    ret["layers_tx"] = np.zeros((ninc, nlayer), dtype=float)

    maxinc = None

    for inc in tqdm.tqdm(incs, disable=not verbose):

        ubar = file[f"/drive/ubar/{inc:d}"][...]
        system.layerSetTargetUbar(ubar)

        u = file[f"/disp/{inc:d}"][...]
        system.setU(u)

        ret["drive_ux"][inc, :] = ubar[drive_x, 0] / ret["height"][drive_x]
        ret["drive_fx"][inc, :] = system.layerFdrive()[drive_x, 0] / ret["kdrive"]
        ret["layers_ux"][inc, :] = system.layerUbar()[:, 0]
        ret["layers_tx"][inc, :] = system.layerTargetUbar()[:, 0]

        Sig = system.Sig() / ret["sig0"]
        Eps = system.Eps() / ret["eps0"]
        idx = system.plastic_CurrentIndex().astype(int)[:, 0].reshape(-1, ret["N"])

        if not system.boundcheck_right(5):
            maxinc = inc
            break

        for i in range(nlayer):
            e = system.layerElements(i)
            E = np.average(Eps[e, ...], weights=dV[e, ...], axis=(0, 1))
            S = np.average(Sig[e, ...], weights=dV[e, ...], axis=(0, 1))
            ret["epsd_layers"][inc, i] = GMat.Epsd(E)
            ret["sigd_layers"][inc, i] = GMat.Sigd(S)

        ret["S_layers"][inc, ret["is_plastic"]] = np.sum(idx - idx_n, axis=1)
        ret["A_layers"][inc, ret["is_plastic"]] = np.sum(idx != idx_n, axis=1)
        ret["epsd"][inc] = GMat.Epsd(np.average(Eps, weights=dV, axis=(0, 1)))
        ret["sigd"][inc] = GMat.Sigd(np.average(Sig, weights=dV, axis=(0, 1)))

        idx_n = np.array(idx, copy=True)

    if maxinc:
        trucate = [
            "epsd",
            "sigd",
            "epsd_layers",
            "sigd_layers",
            "S_layers",
            "A_layers",
            "drive_ux",
            "drive_fx",
            "layers_ux",
            "layers_tx",
        ]
        for key in trucate:
            ret[key] = ret[key][:maxinc, ...]

    ret["steadystate"] = System.steadystate(**ret)

    return ret


def cli_plot(cli_args=None):
    """
    Plot basic output.
    """

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    docstring = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    progname = entry_points[funcname]

    if cli_args is None:
        cli_args = sys.argv[1:]
    else:
        cli_args = [str(arg) for arg in cli_args]

    class MyFormatter(
        argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
    ):
        pass

    parser = argparse.ArgumentParser(
        formatter_class=MyFormatter, description=replace_entry_point(docstring)
    )

    parser.add_argument("-o", "--output", type=str, help="Output file")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("file", type=str, help="File to read")

    args = parser.parse_args(cli_args)

    assert os.path.isfile(os.path.realpath(args.file))

    if args.output:
        if not args.force:
            if os.path.isfile(args.output):
                if not click.confirm(f'Overwrite "{args.output}"?'):
                    raise OSError("Cancelled")

    with h5py.File(args.file, "r") as file:
        system = System.init(file)
        out = basic_output(system, file, verbose=False)

    fig, ax = plt.subplots()
    ax.plot(out["epsd"], out["sigd"])

    if args.output:
        fig.savefig(args.output)
    else:
        plt.show()

    plt.close(fig)


def cli_ensembleinfo(cli_args=None):
    """
    Read information (avalanche size, stress, strain, ...) of an ensemble, and combine into
    a single output file.
    """

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    docstring = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    progname = entry_points[funcname]
    output = file_defaults[funcname]

    if cli_args is None:
        cli_args = sys.argv[1:]
    else:
        cli_args = [str(arg) for arg in cli_args]

    class MyFormatter(
        argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
    ):
        pass

    parser = argparse.ArgumentParser(
        formatter_class=MyFormatter, description=replace_entry_point(docstring)
    )

    parser.add_argument("-o", "--output", type=str, default=output, help="Output file")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("files", nargs="*", type=str, help="Files to read")

    args = parser.parse_args(cli_args)

    assert len(args.files) > 0
    assert np.all([os.path.isfile(os.path.realpath(file)) for file in args.files])
    files = [os.path.relpath(file, os.path.dirname(args.output)) for file in args.files]
    seeds = []

    if not args.force:
        if os.path.isfile(args.output):
            if not click.confirm(f'Overwrite "{args.output}"?'):
                raise OSError("Cancelled")

    fields_norm = [
        "l0",
        "G",
        "K",
        "rho",
        "cs",
        "cd",
        "sig0",
        "eps0",
        "N",
        "t0",
        "dt",
        "kdrive",
        "nlayer",
        "is_plastic",
        "height",
    ]

    fields_full = [
        "epsd",
        "sigd",
        "epsd_layers",
        "sigd_layers",
        "S_layers",
        "A_layers",
        "drive_ux",
        "drive_fx",
        "layers_ux",
        "layers_tx",
        "inc",
        "steadystate",
        "delta_gamma",  # todo: could be moved to normalisation
        "version",
        "dependencies",
    ]

    if os.path.exists(args.output):
        os.remove(args.output)

    for i, (filename, filepath) in enumerate(zip(tqdm.tqdm(files), args.files)):

        with h5py.File(filepath, "r") as file:

            # (re)initialise system
            if i == 0:
                system = System.init(file)
            else:
                system.reset_epsy(System.read_epsy(file))

            # read output
            out = basic_output(system, file, verbose=False)
            seeds += [out["seed"]]

            # store/check normalisation
            if i == 0:
                norm = {key: out[key] for key in fields_norm}
            else:
                for key in fields_norm:
                    if not np.allclose(norm[key], out[key]):
                        raise OSError(f"Inconsistent '{key}'")

            # write full output
            with h5py.File(args.output, "a") as output:
                for key in fields_full:
                    output[f"/full/{filename}/{key}"] = out[key]

    # write normalisation and global info
    with h5py.File(args.output, "a") as output:
        for key, value in norm.items():
            output[f"/normalisation/{key}"] = value

        output["full"].attrs["stored"] = files
        output["full"].attrs["seeds"] = seeds
        meta = output.create_group(f"/meta/{config}/{progname}")
        meta.attrs["version"] = version
        meta.attrs["dependencies"] = dependencies(model)


def view_paraview(
    system: model.System, file: h5py.File, outbasename: str, verbose: bool = True
) -> dict:
    """
    Write ParaView file

    :param system: The system (modified: all increments visited).
    :param file: Open simulation HDF5 archive (read-only).
    :param outbasename: Basename of the output files (appended ".h5" and ".xdmf")
    :param verbose: Print progress.
    """

    assert not os.path.isfile(os.path.realpath(f"{outbasename}.h5"))
    assert not os.path.isfile(os.path.realpath(f"{outbasename}.xdmf"))

    with h5py.File(f"{outbasename}.h5", "w") as output:

        dV = system.quad().AsTensor(2, system.quad().dV())
        sig0 = file["/meta/normalisation/sig"][...]
        eps0 = file["/meta/normalisation/eps"][...]

        output["/coor"] = system.coor()
        output["/conn"] = system.conn()

        series = xh.TimeSeries()

        for inc in tqdm.tqdm(file["/stored"][...], disable=not verbose):

            system.layerSetTargetUbar(file[f"/drive/ubar/{inc:d}"][...])

            u = file[f"/disp/{inc:d}"][...]
            system.setU(u)

            Sig = GMat.Sigd(np.average(system.Sig() / sig0, weights=dV, axis=1))
            Epsp = np.zeros_like(Sig)
            Epsp[system.plastic()] = np.mean(system.plastic_Epsp() / eps0, axis=1)

            output[f"/disp/{inc:d}"] = xh.as3d(u)
            output[f"/sigd/{inc:d}"] = Sig
            output[f"/epsp/{inc:d}"] = Epsp

            series.push_back(
                xh.Unstructured(output, "/coor", "/conn", "Quadrilateral"),
                xh.Attribute(output, f"/disp/{inc:d}", "Node", name="Displacement"),
                xh.Attribute(output, f"/sigd/{inc:d}", "Cell", name="Stress"),
                xh.Attribute(output, f"/epsp/{inc:d}", "Cell", name="Plastic strain"),
            )

        xh.write(series, f"{outbasename}.xdmf")


def cli_view_paraview(cli_args=None):
    """
    Create files to view with ParaView.
    """

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    docstring = textwrap.dedent(inspect.getdoc(globals()[funcname]))

    if cli_args is None:
        cli_args = sys.argv[1:]
    else:
        cli_args = [str(arg) for arg in cli_args]

    class MyFormatter(
        argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
    ):
        pass

    parser = argparse.ArgumentParser(
        formatter_class=MyFormatter, description=replace_entry_point(docstring)
    )

    parser.add_argument(
        "-p",
        "--prefix",
        type=str,
        default="paraview_",
        help="Prefix files to create dedicated Paraview files.",
    )

    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("files", nargs="*", type=str, help="Files to read")

    args = parser.parse_args(cli_args)

    assert len(args.files) > 0
    assert np.all([os.path.isfile(os.path.realpath(file)) for file in args.files])

    if not args.force:

        overwrite = False

        for filepath in args.files:
            dirname, filename = os.path.split(filepath)
            outbasename = f"{args.prefix:s}{os.path.splitext(filename)[0]:s}"
            outbasename = os.path.join(dirname, outbasename)
            if os.path.exists(f"{outbasename}.h5"):
                overwrite = True
                break
            if os.path.exists(f"{outbasename}.xdmf"):
                overwrite = True
                break

        if overwrite:
            if not click.confirm("Overwrite existing output?"):
                raise OSError("Cancelled")

    for i, filepath in enumerate(tqdm.tqdm(args.files)):

        with h5py.File(filepath, "r") as file:

            if i == 0:
                system = System.init(file)
            else:
                system.reset_epsy(System.read_epsy(file))

            dirname, filename = os.path.split(filepath)
            outbasename = f"{args.prefix:s}{os.path.splitext(filename)[0]:s}"
            outbasename = os.path.join(dirname, outbasename)
            view_paraview(system, file, outbasename, verbose=False)
