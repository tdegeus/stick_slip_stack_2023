import argparse
import os

import FrictionQPotFEM.UniformMultiLayerIndividualDrive2d as model
import GooseFEM
import h5py
import numpy as np

from . import mesh
from . import storage
from . import system
from . import tag
from ._version import version

config = "FixedLever"


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

    if delta_gamma is not None:
        delta_gamma = 0.005 * eps0 * np.ones(2000) / k_drive
        delta_gamma[0] = 0

    c = 1.0
    G = 1.0
    K = 4.5 * G  # consistent with PMMA
    rho = G / c ** 2.0
    qL = 2.0 * np.pi / L
    qh = 2.0 * np.pi / h
    alpha = np.sqrt(2.0) * qL * c * rho

    dt = (1.0 / (c * qh)) / 10.0

    with h5py.File(filename, "w") as data:

        storage.dump_with_atttrs(
            data,
            "/coor",
            coor,
            desc="Nodal coordinates [nnode, ndim]",
        )

        storage.dump_with_atttrs(
            data,
            "/conn",
            conn,
            desc="Connectivity (Quad4: nne = 4) [nelem, nne]",
        )

        storage.dump_with_atttrs(
            data,
            "/dofs",
            dofs,
            desc="DOFs per node, accounting for semi-periodicity [nnode, ndim]",
        )

        storage.dump_with_atttrs(
            data,
            "/iip",
            iip,
            desc="Prescribed DOFs [nnp]",
        )

        storage.dump_with_atttrs(
            data,
            "/run/epsd/kick",
            eps0 * 1e-4,
            desc="Strain kick to apply",
        )

        storage.dump_with_atttrs(
            data,
            "/run/dt",
            dt,
            desc="Time step",
        )

        storage.dump_with_atttrs(
            data,
            "/rho",
            rho * np.ones(nelem),
            desc="Mass density [nelem]",
        )

        storage.dump_with_atttrs(
            data,
            "/damping/alpha",
            alpha * np.ones(nelem),
            desc="Damping coefficient (density) [nelem]",
        )

        storage.dump_with_atttrs(
            data,
            "/elastic/elem",
            elastic,
            desc="Elastic elements [nelem - N]",
        )

        storage.dump_with_atttrs(
            data,
            "/elastic/K",
            K * np.ones(len(elastic)),
            desc="Bulk modulus for elements in '/elastic/elem' [nelem - N]",
        )

        storage.dump_with_atttrs(
            data,
            "/elastic/G",
            G * np.ones(len(elastic)),
            desc="Shear modulus for elements in '/elastic/elem' [nelem - N]",
        )

        storage.dump_with_atttrs(
            data,
            "/cusp/elem",
            plastic,
            desc="Plastic elements with cusp potential [nplastic]",
        )

        storage.dump_with_atttrs(
            data,
            "/cusp/K",
            K * np.ones(len(plastic)),
            desc="Bulk modulus for elements in '/cusp/elem' [nplastic]",
        )

        storage.dump_with_atttrs(
            data,
            "/cusp/G",
            G * np.ones(len(plastic)),
            desc="Shear modulus for elements in '/cusp/elem' [nplastic]",
        )

        storage.dump_with_atttrs(
            data,
            "/cusp/epsy/initstate",
            initstate,
            desc="State to use to initialise prrng::pcg32",
        )

        storage.dump_with_atttrs(
            data,
            "/cusp/epsy/initseq",
            initseq,
            desc="Sequence to use to initialise prrng::pcg32",
        )

        storage.dump_with_atttrs(
            data,
            "/cusp/epsy/k",
            k,
            desc="Shape factor of Weibull distribution",
        )

        storage.dump_with_atttrs(
            data,
            "/cusp/epsy/eps0",
            eps0,
            desc="Yield strain normalisation: multiply all yield strains with twice this factor",
        )

        storage.dump_with_atttrs(
            data,
            "/cusp/epsy/eps_offset",
            eps_offset,
            desc="Yield strain offset: add this (small) offset to each yield strain, after normalisation!",
        )

        storage.dump_with_atttrs(
            data,
            "/cusp/epsy/nchunk",
            nchunk,
            desc="Chunk size",
        )

        storage.dump_with_atttrs(
            data,
            "/meta/normalisation/N",
            N,
            desc="Number of blocks along each plastic layer",
        )

        storage.dump_with_atttrs(
            data,
            "/meta/normalisation/l",
            h,
            desc="Elementary block size",
        )

        storage.dump_with_atttrs(
            data,
            "/meta/normalisation/rho",
            rho,
            desc="Elementary density",
        )

        storage.dump_with_atttrs(
            data,
            "/meta/normalisation/G",
            G,
            desc="Uniform shear modulus == 2 mu",
        )

        storage.dump_with_atttrs(
            data,
            "/meta/normalisation/K",
            K,
            desc="Uniform bulk modulus == kappa",
        )

        storage.dump_with_atttrs(
            data,
            "/meta/normalisation/eps",
            eps0,
            desc="Typical yield strain",
        )

        storage.dump_with_atttrs(
            data,
            "/meta/normalisation/sig",
            2.0 * G * eps0,
            desc="== 2 G eps0",
        )

        storage.dump_with_atttrs(
            data,
            "/meta/seed_base",
            seed,
            desc="Basic seed == 'unique' identifier",
        )

        storage.dump_with_atttrs(
            data,
            f"/meta/Run{config}/version",
            version,
            desc="Version when generating",
        )

        elemmap = stitch.elemmap()
        nodemap = stitch.nodemap()

        storage.dump_with_atttrs(
            data,
            "/layers/stored",
            np.arange(len(elemmap)).astype(np.int64),
            desc="Layers in simulation",
        )

        for i in range(len(elemmap)):
            data[f"/layers/{i:d}/elemmap"] = elemmap[i]
            data[f"/layers/{i:d}/nodemap"] = nodemap[i]

        storage.dump_with_atttrs(
            data,
            "/layers/is_plastic",
            is_plastic,
            desc="Per layer: true is the layer is plastic",
        )

        storage.dump_with_atttrs(
            data,
            "/drive/k",
            k_drive,
            desc="Stiffness of the spring providing the drive",
        )

        storage.dump_with_atttrs(
            data,
            "/drive/symmetric",
            symmetric,
            desc="If false, the driving spring buckles under tension.",
        )

        storage.dump_with_atttrs(
            data,
            "/drive/drive",
            drive,
            desc="Per layer: true when the layer's mean position is actuated",
        )

        storage.dump_with_atttrs(
            data,
            "/drive/delta_gamma",
            delta_gamma,
            desc="Affine simple shear increment",
        )

        storage.dump_with_atttrs(
            data,
            "/drive/height",
            Hi,
            desc="Height of the loading frame of each layer",
        )


def run(filename: str, dev: bool):
    """
    Run the simulation.

    :param filename: Name of the input/output file (appended).
    :param dev: If ``True`` uncommitted changes are allowed.
    """

    basename = os.path.basename(filename)

    with h5py.File(filename, "a") as data:

        this = system.initsystem(data)

        # check version compatibility

        assert dev or not tag.has_uncommited(version)
        assert dev or not tag.any_has_uncommited(model.version_dependencies())

        path = f"/meta/Run{config}/version"
        if version != "None":
            if path in data:
                assert tag.greater_equal(version, str(data[path].asstr()[...]))
            else:
                data[path] = version

        path = f"/meta/Run{config}/version_dependencies"
        if path in data:
            assert tag.all_greater_equal(
                model.version_dependencies(), data[path].asstr()[...]
            )
        else:
            data[path] = model.version_dependencies()

        if f"/meta/Run{config}/completed" in data:
            print("Marked completed, skipping")
            return 1

        # restore or initialise the this / output

        if "/stored" in data:

            inc = int(data["/stored"][-1])
            this.setT(data["/t"][inc])
            this.setU(data[f"/disp/{inc:d}"][...])
            this.layerSetTargetUbar(data[f"/drive/ubar/{inc:d}"][...])
            print(f'"{basename}": Loading, inc = {inc:d}')

        else:

            inc = int(0)

            dset = data.create_dataset(
                "/stored", (1,), maxshape=(None,), dtype=np.uint64
            )
            dset[0] = inc
            dset.attrs[
                "desc"
            ] = 'List of increments in "/disp/{:d}" and "/drive/ubar/{0:d}"'

            dset = data.create_dataset("/t", (1,), maxshape=(None,), dtype=np.float64)
            dset[0] = this.t()
            dset.attrs["desc"] = "Per increment: time at the end of the increment"

            data[f"/disp/{inc}"] = this.u()
            data[f"/disp/{inc}"].attrs[
                "desc"
            ] = "Displacement (at the end of the increment)."
            data[f"/drive/ubar/{inc}"] = this.layerTargetUbar()
            data[f"/drive/ubar/{inc}"].attrs[
                "desc"
            ] = "Position of the loading frame of each layer."

        # run

        height = data["/drive/height"][...]
        delta_gamma = data["/drive/delta_gamma"][...]

        assert np.isclose(delta_gamma[0], 0.0)
        inc += 1

        for dgamma in delta_gamma[inc:]:

            this.layerTagetUbar_addAffineSimpleShear(dgamma, height)
            niter = this.minimise()
            print(f'"{basename}": inc = {inc:8d}, niter = {niter:8d}')

            storage.dset_extend1d(data, "/stored", inc, inc)
            storage.dset_extend1d(data, "/t", inc, this.t())
            data[f"/disp/{inc:d}"] = this.u()
            data[f"/drive/ubar/{inc:d}"] = this.layerTargetUbar()

            inc += 1

        data["/meta/RunFixedLever/completed"] = 1


def cli_run():

    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="Input file (read/write)")
    parser.add_argument("--dev", action="store_true", help="Allow uncommited changes")
    parser.add_argument("-v", "--version", action="version", version=version)
    args = parser.parse_args()

    assert os.path.isfile(os.path.realpath(args.file))

    run(args.file, args.dev)
