import argparse
import FrictionQPotFEM.UniformMultiLayerIndividualDrive2d as model
import h5py
import os
import GooseFEM
import numpy as np
import unittest

from . import system
from . import tag
from . import mesh
from ._version import version

def dset_extend1d(data: h5py.File, key: str, i: int, value):

    dset = data[key]
    if dset.size <= i:
        dset.resize((i + 1, ))
    dset[i] = value


def mysave(myfile, key, data, **kwargs):
    myfile[key] = data
    for attr in kwargs:
        myfile[key].attrs[attr] = kwargs[attr]


def generate(filename: str, N: int, nplates: int, seed: int, k_drive: float, symmetric = True: bool, delta_gamma: float=None):
    """
    Generate an input file.

    :param filename: Filename of the input file (overwritten).
    :param N: Number of blocks in x-direction (the height of a plate == N /4).
    :param seed: Base seed to use to generate the disorder.
    :param k_drive: Stiffness of the drive string.
    :param symmetric: Set the drive string symmetric.
    :param delta_gamma: Loading history (for testing only).
    """

    M = int(N / 4)
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

        mysave(
            data,
            "/coor",
            coor,
            desc="Nodal coordinates [nnode, ndim]",
        )

        mysave(
            data,
            "/conn",
            conn,
            desc="Connectivity (Quad4: nne = 4) [nelem, nne]",
        )

        mysave(
            data,
            "/dofs",
            dofs,
            desc="DOFs per node, accounting for semi-periodicity [nnode, ndim]",
        )

        mysave(
            data,
            "/iip",
            iip,
            desc="Prescribed DOFs [nnp]",
        )

        mysave(
            data,
            "/run/epsd/kick",
            eps0 * 1e-4,
            desc="Strain kick to apply",
        )

        mysave(
            data,
            "/run/dt",
            dt,
            desc="Time step",
        )

        mysave(
            data,
            "/rho",
            rho * np.ones(nelem),
            desc="Mass density [nelem]",
        )

        mysave(
            data,
            "/damping/alpha",
            alpha * np.ones(nelem),
            desc="Damping coefficient (density) [nelem]",
        )

        mysave(
            data,
            "/cusp/elem",
            plastic,
            desc="Plastic elements with cusp potential [nplastic]",
        )

        mysave(
            data,
            "/cusp/K",
            K * np.ones(len(plastic)),
            desc="Bulk modulus for elements in '/cusp/elem' [nplastic]",
        )

        mysave(
            data,
            "/cusp/G",
            G * np.ones(len(plastic)),
            desc="Shear modulus for elements in '/cusp/elem' [nplastic]",
        )

        mysave(
            data,
            "/cusp/epsy/initstate",
            initstate,
            desc="State to use to initialise prrng::pcg32",
        )

        mysave(
            data,
            "/cusp/epsy/initseq",
            initseq,
            desc="Sequence to use to initialise prrng::pcg32",
        )

        mysave(
            data,
            "/cusp/epsy/k",
            k,
            desc="Shape factor of Weibull distribution",
        )

        mysave(
            data,
            "/cusp/epsy/eps0",
            eps0,
            desc="Yield strain normalisation: multiply all yield strains with twice this factor",
        )

        mysave(
            data,
            "/cusp/epsy/eps_offset",
            eps_offset,
            desc="Yield strain offset: add this (small) offset to each yield strain, after normalisation!",
        )

        mysave(
            data,
            "/cusp/epsy/nchunk",
            nchunk,
            desc="Chunk size",
        )

        mysave(
            data,
            "/elastic/elem",
            elastic,
            desc="Elastic elements [nelem - N]",
        )

        mysave(
            data,
            "/elastic/K",
            K * np.ones(len(elastic)),
            desc="Bulk modulus for elements in '/elastic/elem' [nelem - N]",
        )

        mysave(
            data,
            "/elastic/G",
            G * np.ones(len(elastic)),
            desc="Shear modulus for elements in '/elastic/elem' [nelem - N]",
        )

        mysave(
            data,
            "/meta/normalisation/N",
            N,
            desc="Number of blocks along each plastic layer",
        )

        mysave(
            data,
            "/meta/normalisation/l",
            h,
            desc="Elementary block size",
        )

        mysave(
            data,
            "/meta/normalisation/rho",
            rho,
            desc="Elementary density",
        )

        mysave(
            data,
            "/meta/normalisation/G",
            G,
            desc="Uniform shear modulus == 2 mu",
        )

        mysave(
            data,
            "/meta/normalisation/K",
            K,
            desc="Uniform bulk modulus == kappa",
        )

        mysave(
            data,
            "/meta/normalisation/eps",
            eps0,
            desc="Typical yield strain",
        )

        mysave(
            data,
            "/meta/normalisation/sig",
            2.0 * G * eps0,
            desc="== 2 G eps0",
        )

        mysave(
            data,
            "/meta/seed_base",
            seed,
            desc="Basic seed == 'unique' identifier",
        )

        mysave(
            data,
            f"/meta/{basename}/{genscript}",
            myversion,
            desc="Version when generating",
        )

        elemmap = stitch.elemmap()
        nodemap = stitch.nodemap()

        mysave(
            data,
            "/layers/stored",
            np.arange(len(elemmap)).astype(np.int64),
            desc="Layers in simulation",
        )

        for i in range(len(elemmap)):
            data[f"/layers/{i:d}/elemmap"] = elemmap[i]
            data[f"/layers/{i:d}/nodemap"] = nodemap[i]

        mysave(
            data,
            "/layers/is_plastic",
            is_plastic,
            desc="Per layer: true is the layer is plastic",
        )

        mysave(
            data,
            "/drive/k",
            k_drive,
            desc="Stiffness of the spring providing the drive",
        )

        mysave(
            data,
            "/drive/symmetric",
            symmetric,
            desc="If false, the driving spring buckles under tension.",
        )

        mysave(
            data,
            "/drive/drive",
            drive,
            desc="Per layer: true when the layer's mean position is actuated",
        )

        mysave(
            data,
            "/drive/delta_gamma",
            delta_gamma,
            desc="Affine simple shear increment",
        )

        mysave(
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

        path = "/meta/RunFixedLever/version"
        if version != "None":
            if path in data:
                assert tag.greater_equal(version, str(data[path].asstr()[...]))
            else:
                data[path] = version

        path = "/meta/RunFreeLever/version_dependencies"
        if path in data:
            assert tag.all_greater_equal(model.version_dependencies(), data[path].asstr()[...])
        else:
            data[path] = model.version_dependencies()

        if "/meta/RunFixedLever/completed" in data:
            print("Marked completed, skipping")
            return 1

        # restore or initialise the this / output

        if "/stored" in data:

            inc = int(data["/stored"][-1])
            this.setT(data["/t"][inc])
            this.setU(data[f"/disp/{inc:d}"][...])
            this.layerSetTargetUbar(data[f"/drive/ubar/{inc:d}"][...])
            print(f"\"{basename}\": Loading, inc = {inc:d}")

        else:

            inc = int(0)

            dset = data.create_dataset("/stored", (1, ), maxshape=(None, ), dtype=np.uint64)
            dset[0] = inc
            dset.attrs["desc"] = "List of increments in \"/disp/{:d}\" and \"/drive/ubar/{0:d}\""

            dset = data.create_dataset("/t", (1, ), maxshape=(None, ), dtype=np.float64)
            dset[0] = this.t()
            dset.attrs["desc"] = "Per increment: time at the end of the increment"

            data[f"/disp/{inc}"] = this.u()
            data[f"/disp/{inc}"].attrs["desc"] = "Displacement (at the end of the increment)."
            data[f"/drive/ubar/{inc}"] = this.layerTargetUbar()
            data[f"/drive/ubar/{inc}"].attrs["desc"] = "Position of the loading frame of each layer."

        # run

        height = data["/drive/height"][...]
        delta_gamma = data["/drive/delta_gamma"][...]

        assert np.isclose(delta_gamma[0], 0.0)
        inc += 1

        for dgamma in delta_gamma[inc:]:

            this.layerTagetUbar_addAffineSimpleShear(dgamma, height)
            niter = this.minimise()
            print(f"\"{basename}\": inc = {inc:8d}, niter = {niter:8d}")

            dset_extend1d(data, "/stored", inc, inc)
            dset_extend1d(data, "/t", inc, this.t())
            data[f"/disp/{inc:d}"] = this.u()
            data[f"/drive/ubar/{inc:d}"] = this.layerTargetUbar()

            inc += 1

        data["/meta/RunFixedLever/completed"] = 1


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="Input file (read/write)")
    parser.add_argument("--dev", action="store_true", help="Allow uncommited changes")
    parser.add_argument("-v", "--version", action="version", version=version)
    args = parser.parse_args()

    assert os.path.isfile(os.path.realpath(args.file))

    run(args.file, args.dev)


class MyTests(unittest.TestCase):

    def test_run(self):

        generate("mytest.h5", 9, 2, 0, 1e-3, 1, delta_gamma=[0, 0.005, 0.01])

        # self.assertTrue(True)

    # def test_all_greater_equal(self):

    #     a = ["xtensor=3.2.1", "frictionqpotfem=4.4"]
    #     b = ["xtensor=3.2.0", "frictionqpotfem=4.4", "eigen=3.0.0"]
    #     self.assertTrue(all_greater_equal(a, b))

if __name__ == "__main__":

    unittest.main()





