import argparse
import inspect
import os
import sys
import textwrap

import click
import FrictionQPotFEM.UniformMultiLayerIndividualDrive2d as model
import GMatElastoPlasticQPot.Cartesian2d as GMat
import h5py
import numpy as np
import shelephant
import tqdm
import XDMFWrite_h5py as xh

from . import slurm
from . import System
from ._version import version

config = "FixedLever"

entry_points = dict(
    cli_ensembleinfo="FixedLever_EnsembleInfo",
    cli_find_completed="FixedLever_FindCompleted",
    cli_generate="FixedLever_Generate",
    cli_job_rerun_multislip="FixedLever_EventsJob",
    cli_plot="FixedLever_Plot",
    cli_rerun_event="FixedLever_Events",
    cli_run="FixedLever_Run",
    cli_view_paraview="FixedLever_Paraview",
)

file_defaults = dict(
    cli_ensembleinfo="EnsembleInfo.h5",
    cli_find_completed="completed.yaml",
    cli_rerun_event="FixedLever_Events.h5",
)


def replace_entry_point(doc):
    """
    Replace ":py:func:`...`" with the relevant entry_point name
    """
    for ep in entry_points:
        doc = doc.replace(fr":py:func:`{ep:s}`", entry_points[ep])
    return doc


def init(file: h5py.File) -> model.System:
    """
    Initialise system from file.

    :param file: Open simulation HDF5 archive (read-only).
    :return: The initialised system.
    """
    return System.init(file, model)


def generate(**kwargs):
    """
    See :py:func:`System.generate`, skipping the arguments ``config`` and ``progname``.
    """
    return System.generate(**kwargs, config=config, progname=entry_points["cli_generate"])


def run(**kwargs):
    """
    See :py:func:`System.run`,
    skipping the arguments ``config``, ``progname``, ``model``, and ``init_function``.
    """
    return System.run(
        **kwargs, config=config, progname=entry_points["cli_run"], model=model, init_function=init
    )


def basic_output(*args, **kwargs):
    """
    See :py:func:`System.basic_output`,
    skipping the arguments ``config``, ``progname``, ``model``, and ``init_function``.
    """
    return System.basic_output(*args, **kwargs)


def cli_generate(cli_args=None):
    """
    See :py:func:`System.cli_generate`.
    """
    return System.cli_generate(cli_args=cli_args, entry_points=entry_points, config=config)


def cli_run(cli_args=None):
    """
    See :py:func:`System.cli_run`.
    """
    return System.cli_run(
        cli_args=cli_args, entry_points=entry_points, config=config, model=model, init_function=init
    )


def find_completed(filepaths: list[str]) -> list[str]:
    """
    List simulations marked completed.

    :param filepaths: List of files.
    :return: Those entries in ``filepaths`` that correspond to completed files.
    """

    progname = entry_points["cli_run"]
    completed = []

    for filepath in filepaths:
        with h5py.File(filepath, "r") as file:
            if "completed" in file[f"/meta/{config}/{progname}"].attrs:
                completed.append(filepath)

    return completed


def cli_find_completed(cli_args=None):
    """
    List simulations marked completed.
    """

    class MyFmt(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=replace_entry_point(doc))

    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite")
    parser.add_argument("-o", "--output", default=file_defaults[funcname], help="Output file")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("files", type=str, nargs="*", help="Simulation files")

    if cli_args is None:
        args = parser.parse_args(sys.argv[1:])
    else:
        args = parser.parse_args([str(arg) for arg in cli_args])

    assert np.all([os.path.isfile(file) for file in args.files])

    completed = find_completed(args.files)
    shelephant.yaml.dump(args.output, completed, force=args.force)

    if cli_args is not None:
        return completed


def runinc_event_basic(system: model.System, file: h5py.File, inc: int, Smax=sys.maxsize) -> dict:
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

    class MyFmt(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=replace_entry_point(doc))
    progname = entry_points[funcname]
    output = file_defaults[funcname]

    parser.add_argument(
        "-s",
        "--smax",
        type=int,
        help="Truncate at a given maximal total S",
    )

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
        system = init(file)
        if args.smax is None:
            ret = runinc_event_basic(system, file, args.inc)
        else:
            ret = runinc_event_basic(system, file, args.inc, args.smax)

    with h5py.File(args.output, "w") as file:
        file["r"] = ret["r"]
        file["t"] = ret["t"]
        meta = file.create_group(f"/meta/{config}/{progname}")
        meta.attrs["version"] = version
        meta.attrs["dependencies"] = System.dependencies(model)


def cli_job_rerun_multislip(cli_args=None):
    """
    Rerun increments that have events in which more than one layer slips.
    """

    class MyFmt(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=replace_entry_point(doc))

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

    if cli_args is None:
        args = parser.parse_args(sys.argv[1:])
    else:
        args = parser.parse_args([str(arg) for arg in cli_args])

    assert os.path.isfile(args.info)
    assert os.path.isdir(args.outdir)

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
                commands += [f"{executable} -i {i:d} -s {s:d} -o {simid}_inc={i:d}.h5 {relfile}"]

    slurm.serial_group(
        commands,
        basename=args.executable.replace(" ", "_"),
        group=args.group,
        outdir=args.outdir,
        sbatch={"time": args.time},
    )


def cli_plot(cli_args=None):
    """
    Plot basic output.
    """
    return System.cli_plot(cli_args=cli_args, init_function=init)


def cli_ensembleinfo(cli_args=None):
    """
    Read information (avalanche size, stress, strain, ...) of an ensemble, and combine into
    a single output file.
    """

    class MyFmt(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=replace_entry_point(doc))
    progname = entry_points[funcname]
    output = file_defaults[funcname]

    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite")
    parser.add_argument("-o", "--output", type=str, default=output, help="Output file")
    parser.add_argument("-v", "--version", action="version", version=version)
    parser.add_argument("files", nargs="*", type=str, help="Files to read")

    if cli_args is None:
        args = parser.parse_args(sys.argv[1:])
    else:
        args = parser.parse_args([str(arg) for arg in cli_args])

    assert len(args.files) > 0
    assert np.all([os.path.isfile(file) for file in args.files])
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
    ]

    if os.path.exists(args.output):
        os.remove(args.output)

    for i, (filename, filepath) in enumerate(zip(tqdm.tqdm(files), args.files)):

        with h5py.File(filepath, "r") as file:

            # (re)initialise system
            if i == 0:
                system = init(file)
            else:
                system.reset_epsy(System.read_epsy(file))

            # read output
            out = System.basic_output(system, file, verbose=False)
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
        meta.attrs["dependencies"] = System.dependencies(model)


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

    assert not os.path.isfile(f"{outbasename}.h5")
    assert not os.path.isfile(f"{outbasename}.xdmf")

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

    class MyFmt(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass

    funcname = inspect.getframeinfo(inspect.currentframe()).function
    doc = textwrap.dedent(inspect.getdoc(globals()[funcname]))
    parser = argparse.ArgumentParser(formatter_class=MyFmt, description=replace_entry_point(doc))

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

    if cli_args is None:
        args = parser.parse_args(sys.argv[1:])
    else:
        args = parser.parse_args([str(arg) for arg in cli_args])

    assert len(args.files) > 0
    assert np.all([os.path.isfile(file) for file in args.files])

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
                system = init(file)
            else:
                system.reset_epsy(System.read_epsy(file))

            dirname, filename = os.path.split(filepath)
            outbasename = f"{args.prefix:s}{os.path.splitext(filename)[0]:s}"
            outbasename = os.path.join(dirname, outbasename)
            view_paraview(system, file, outbasename, verbose=False)
