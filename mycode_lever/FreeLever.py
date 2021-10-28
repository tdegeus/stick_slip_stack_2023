import FrictionQPotFEM.UniformMultiLayerLeverDrive2d as model
import h5py

from . import System

config = "FreeLever"

entry_points = dict(
    cli_copy_perturbation="FreeLever_CopyPerturbation",
    cli_generate="FreeLever_Generate",
    cli_plot="FreeLever_Plot",
    cli_run="FreeLever_Run",
)

file_defaults = dict()


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
    system = System.init(file, model)
    system.setLeverProperties(file["/drive/H"][...], file["/drive/height"][...])
    return system


def generate(*args, **kwargs):
    """
    See :py:func:`System.generate`.
    """
    return System.generate(*args, **kwargs, config=config, progname=entry_points["cli_generate"])


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


def cli_copy_perturbation(cli_args=None):
    """
    See :py:func:`System.cli_copy_perturbation`.
    """
    return System.cli_copy_perturbation(cli_args=cli_args, entry_points=entry_points, config=config)


def basic_output(*args, **kwargs):
    """
    See :oy:func:`System.basic_output`.
    """
    return System.basic_output(*args, **kwargs)


def cli_plot(cli_args=None):
    """
    Plot basic output.
    """
    return System.cli_plot(cli_args=cli_args, init_function=init)
