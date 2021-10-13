import argparse
import itertools
import os

import GooseFEM
import h5py
import numpy as np
import setuptools_scm

# import prrng

# ==================================================================================================


class MyMesh:
    def coor(self):
        return self.m_coor

    def conn(self):
        return self.m_conn

    def nodesLeftEdge(self):
        return self.m_nodesLeftEdge

    def nodesRightEdge(self):
        return self.m_nodesRightEdge

    def nodesBottomEdge(self):
        return self.m_nodesBottomEdge

    def nodesTopEdge(self):
        return self.m_nodesTopEdge

    def nodesLeftOpenEdge(self):
        return self.m_nodesLeftEdge[1:-1]

    def nodesRightOpenEdge(self):
        return self.m_nodesRightEdge[1:-1]

    def nodesBottomOpenEdge(self):
        return self.m_nodesBottomEdge[1:-1]

    def nodesTopOpenEdge(self):
        return self.m_nodesTopEdge[1:-1]

    def check(self):

        assert np.max(self.m_conn) == self.m_coor.shape[0] - 1
        assert np.all(np.unique(self.m_conn) == np.arange(self.m_coor.shape[0]))
        assert np.all(np.unique(self.m_nodesLeftEdge) == self.m_nodesLeftEdge)
        assert np.all(np.unique(self.m_nodesRightEdge) == self.m_nodesRightEdge)
        assert np.all(np.unique(self.m_nodesBottomEdge) == self.m_nodesBottomEdge)
        assert np.all(np.unique(self.m_nodesTopEdge) == self.m_nodesTopEdge)
        assert np.allclose(self.m_coor[self.m_nodesLeftEdge, 0], 0)
        assert np.allclose(self.m_coor[self.m_nodesRightEdge, 0], self.m_Lx)
        assert np.allclose(self.m_coor[self.m_nodesBottomEdge, 1], 0)
        assert np.allclose(self.m_coor[self.m_nodesTopEdge, 1], self.m_Ly)
        assert self.m_nodesBottomEdge[0] == self.m_nodesLeftEdge[0]
        assert self.m_nodesBottomEdge[-1] == self.m_nodesRightEdge[0]
        assert self.m_nodesTopEdge[0] == self.m_nodesLeftEdge[-1]
        assert self.m_nodesTopEdge[-1] == self.m_nodesRightEdge[-1]

    def plot(self):

        import matplotlib.pyplot as plt
        import GooseMPL as gplt

        plt.style.use(["goose", "goose-latex"])

        fig, ax = plt.subplots()

        gplt.patch(coor=self.coor(), conn=self.conn(), cindex=np.arange(self.conn().shape[0]))

        ax.plot(self.coor()[:, 0], self.coor()[:, 1], marker=".", ls="none")

        ax.plot(
            self.coor()[self.nodesLeftEdge(), 0],
            self.coor()[self.nodesLeftEdge(), 1],
            marker="o",
            ls="none",
        )

        ax.plot(
            self.coor()[self.nodesRightEdge(), 0],
            self.coor()[self.nodesRightEdge(), 1],
            marker="o",
            ls="none",
        )

        ax.plot(
            self.coor()[self.nodesBottomEdge(), 0],
            self.coor()[self.nodesBottomEdge(), 1],
            marker="o",
            ls="none",
        )

        ax.plot(
            self.coor()[self.nodesTopEdge(), 0],
            self.coor()[self.nodesTopEdge(), 1],
            marker="o",
            ls="none",
        )

        plt.show()


# ==================================================================================================


class LayerElastic(MyMesh):
    def __init__(self, Nx, Ny, h):

        mesh = GooseFEM.Mesh.Quad4.FineLayer(Nx, Ny, h)
        coor = mesh.coor()
        conn = mesh.conn()
        weak = mesh.elementsMiddleLayer()

        bot_conn = conn[: weak[0], :]

        el = int(weak[-1] + 1)
        top_conn = conn[el:, :] - conn[weak[0], 3]

        el = int(conn[weak[-1], 1] + 1)
        bot_coor = coor[:el, :]

        nd = conn[weak[0], 3]
        top_coor = coor[nd:, :]
        top_coor[:, 1] -= np.min(top_coor[:, 1])

        top_ntop = mesh.nodesTopEdge() - conn[weak[0], 3]
        bot_nbot = mesh.nodesBottomEdge()

        self.m_Lx = Nx * h
        self.m_Ly = 2 * np.max(bot_coor[:, 1])

        n = mesh.nodesLeftEdge()
        m = int((n.size - n.size % 2) / 2)
        bot_nlft = n[:m]
        top_nlft = n[m:] - conn[weak[0], 3]

        n = mesh.nodesRightEdge()
        m = int((n.size - n.size % 2) / 2)
        bot_nrgt = n[:m]
        top_nrgt = n[m:] - conn[weak[0], 3]

        stitch = GooseFEM.Mesh.Vstack()
        stitch.push_back(top_coor, top_conn, [0], top_ntop)
        stitch.push_back(bot_coor, bot_conn, bot_nbot, [0])

        self.m_coor = stitch.coor()
        self.m_conn = stitch.conn()
        self.m_nodesLeftEdge = stitch.nodeset([top_nlft, bot_nlft])
        self.m_nodesRightEdge = stitch.nodeset([top_nrgt, bot_nrgt])
        self.m_nodesBottomEdge = stitch.nodeset(np.arange(0, Nx + 1), 0)
        self.m_nodesTopEdge = stitch.nodeset(np.arange(conn[weak[0], 0], conn[weak[-1], 1] + 1), 1)

        self.check()
        assert np.allclose(
            self.m_coor[self.m_nodesLeftEdge, 1], self.m_coor[self.m_nodesRightEdge, 1]
        )
        assert np.allclose(
            self.m_coor[self.m_nodesBottomEdge, 0], self.m_coor[self.m_nodesTopEdge, 0]
        )
        assert self.m_nodesBottomEdge.size == Nx + 1
        assert self.m_nodesTopEdge.size == Nx + 1


# ==================================================================================================


class TopLayerElastic(MyMesh):
    def __init__(self, Nx, Ny, h):

        mesh = GooseFEM.Mesh.Quad4.FineLayer(Nx, Ny, h)

        coor = mesh.coor()
        conn = mesh.conn()
        weak = mesh.elementsMiddleLayer()

        el = int(weak[-1] + 1)
        top_conn = conn[el:, :] - conn[weak[0], 3]

        el = conn[weak[0], 3]
        top_coor = coor[el:, :]
        top_coor[:, 1] -= np.min(top_coor[:, 1])

        top_ntop = mesh.nodesTopEdge() - conn[weak[0], 3]

        n = mesh.nodesLeftEdge()
        m = int((n.size - n.size % 2) / 2)
        top_nlft = n[m:] - conn[weak[0], 3]

        n = mesh.nodesRightEdge()
        m = int((n.size - n.size % 2) / 2)
        top_nrgt = n[m:] - conn[weak[0], 3]

        H = top_coor[top_conn[-1, 2], 0] - top_coor[top_conn[-1, 3], 0]
        Ly = top_coor[top_conn[-1, -1], 1] - top_coor[top_conn[0, 0], 1]

        extra = GooseFEM.Mesh.Quad4.Regular(top_ntop.size - 1, int(np.ceil((Ny * h - Ly) / H)), H)

        stitch = GooseFEM.Mesh.Vstack()
        stitch.push_back(top_coor, top_conn, [0], top_ntop)
        stitch.push_back(extra.coor(), extra.conn(), extra.nodesBottomEdge(), extra.nodesTopEdge())

        self.m_coor = stitch.coor()
        self.m_conn = stitch.conn()
        self.m_nodesLeftEdge = stitch.nodeset([top_nlft, extra.nodesLeftEdge()])
        self.m_nodesRightEdge = stitch.nodeset([top_nrgt, extra.nodesRightEdge()])
        self.m_nodesBottomEdge = stitch.nodeset(np.arange(0, Nx + 1), 0)
        self.m_nodesTopEdge = stitch.nodeset(extra.nodesTopEdge(), 1)

        self.m_Lx = Nx * h
        self.m_Ly = np.max(self.m_coor[:, 1])

        self.check()


# ==================================================================================================


class BottomLayerElastic(MyMesh):
    def __init__(self, Nx, Ny, h):

        mesh = GooseFEM.Mesh.Quad4.FineLayer(Nx, Ny, h)

        coor = mesh.coor()
        conn = mesh.conn()
        weak = mesh.elementsMiddleLayer()
        bot_conn = conn[: weak[0], :]
        bot_coor = coor[: int(conn[weak[-1], 1] + 1), :]
        bot_nbot = mesh.nodesBottomEdge()

        n = mesh.nodesLeftEdge()
        m = int((n.size - n.size % 2) / 2)
        bot_nlft = n[:m]

        n = mesh.nodesRightEdge()
        m = int((n.size - n.size % 2) / 2)
        bot_nrgt = n[:m]

        H = bot_coor[bot_conn[0, 1], 0] - bot_coor[bot_conn[0, 0], 0]
        Ly = bot_coor[bot_conn[-1, -1], 1] - bot_coor[bot_conn[0, 0], 1]

        extra = GooseFEM.Mesh.Quad4.Regular(bot_nbot.size - 1, int(np.ceil((Ny * h - Ly) / H)), H)

        stitch = GooseFEM.Mesh.Vstack()
        stitch.push_back(extra.coor(), extra.conn(), extra.nodesBottomEdge(), extra.nodesTopEdge())
        stitch.push_back(bot_coor, bot_conn, bot_nbot, [0])

        self.m_coor = stitch.coor()
        self.m_conn = stitch.conn()
        self.m_nodesLeftEdge = stitch.nodeset([extra.nodesLeftEdge(), bot_nlft])
        self.m_nodesRightEdge = stitch.nodeset([extra.nodesRightEdge(), bot_nrgt])
        self.m_nodesBottomEdge = stitch.nodeset(extra.nodesBottomEdge(), 0)
        self.m_nodesTopEdge = stitch.nodeset(np.arange(conn[weak[0], 0], conn[weak[-1], 1] + 1), 1)

        self.m_Lx = Nx * h
        self.m_Ly = np.max(self.m_coor[:, 1])

        self.check()


# ==================================================================================================


def mysave(myfile, key, data, **kwargs):
    myfile[key] = data
    for attr in kwargs:
        myfile[key].attrs[attr] = kwargs[attr]


def generate(myversion, filename, N, nplates, seed, rid, k_drive, symmetric):

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

    layer_bot = BottomLayerElastic(N, M, h)
    layer_top = TopLayerElastic(N, M, h)
    layer_elas = LayerElastic(N, M, h)
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

    Hlever = 10.0 * np.diff(Hi)[0]
    delta_gamma = 0.005 * eps0 * np.ones(2000) * Hlever
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

        mysave(
            data,
            "/drive/H",
            Hlever,
            desc="Height of the spring driving the lever",
        )


# ==================================================================================================

if __name__ == "__main__":

    basedir = os.path.dirname(__file__)
    basename = os.path.split(basedir)[1]
    genscript = os.path.splitext(os.path.basename(__file__))[0]
    myversion = setuptools_scm.get_version(root=os.path.join(basedir, ".."))

    parser = argparse.ArgumentParser()
    parser.add_argument("outdir", type=str)
    parser.add_argument("-N", type=int, default=3 ** 6, help="System size")
    parser.add_argument("-n", type=int, default=10, help="Number of systems to generate")
    parser.add_argument("-i", type=int, default=0, help="Simulation index at which to start")
    parser.add_argument("-m", type=int, default=5, help="Maximum number of plastic (minimum == 2)")
    parser.add_argument("--seed", type=int, default=0, help="Base seed")
    parser.add_argument("--max-plates", type=int, default=100, help="Maximum number of plates")
    parser.add_argument("--symmetric", type=int, default=1, help="Drive string symmetric")
    parser.add_argument("-k", type=float, default=1e-3, help="Drive string stiffness")
    args = parser.parse_args()

    for sid, nplates in itertools.product(range(args.i, args.i + args.n), range(2, args.m + 1)):

        filename = "id={:03d}_nplates={:d}_kplate={:.0e}_symmetric={:d}.h5".format(
            sid, nplates, args.k, args.symmetric
        )

        generate(
            myversion,
            os.path.join(args.outdir, filename),
            args.N,
            nplates,
            args.seed + sid * args.N * (args.max_plates - 1),
            sid,
            args.k,
            args.symmetric,
        )
