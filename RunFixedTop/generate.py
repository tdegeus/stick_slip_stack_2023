import GooseFEM
import h5py
import itertools
import numpy as np
import os
import prrng
import setuptools_scm

basename = os.path.split(os.path.dirname(os.path.abspath(__file__)))[1]
genscript = os.path.splitext(os.path.basename(__file__))[0]
myversion = setuptools_scm.get_version(root=os.path.join(os.path.dirname(__file__), '..'))

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
        return self.m_nodesLeftEdge[1: -1]

    def nodesRightOpenEdge(self):
        return self.m_nodesRightEdge[1: -1]

    def nodesBottomOpenEdge(self):
        return self.m_nodesBottomEdge[1: -1]

    def nodesTopOpenEdge(self):
        return self.m_nodesTopEdge[1: -1]

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

        plt.style.use(['goose', 'goose-latex'])

        fig, ax = plt.subplots()

        gplt.patch(coor=self.coor(), conn=self.conn(), cindex=np.arange(self.conn().shape[0]))
        ax.plot(self.coor()[:, 0], self.coor()[:, 1], marker='.', linestyle='none')
        ax.plot(self.coor()[self.nodesLeftEdge(), 0], self.coor()[self.nodesLeftEdge(), 1], marker='o', linestyle='none')
        ax.plot(self.coor()[self.nodesRightEdge(), 0], self.coor()[self.nodesRightEdge(), 1], marker='o', linestyle='none')
        ax.plot(self.coor()[self.nodesBottomEdge(), 0], self.coor()[self.nodesBottomEdge(), 1], marker='o', linestyle='none')
        ax.plot(self.coor()[self.nodesTopEdge(), 0], self.coor()[self.nodesTopEdge(), 1], marker='o', linestyle='none')
        plt.show()

# ==================================================================================================

class LayerElastic(MyMesh):

    def __init__(self, Nx, Ny, h):

        mesh = GooseFEM.Mesh.Quad4.FineLayer(Nx, Ny, h)
        coor = mesh.coor()
        conn = mesh.conn()
        weak = mesh.elementsMiddleLayer()
        bot_conn = conn[:weak[0], :]
        top_conn = conn[int(weak[-1] + 1):, :] - conn[weak[0], 3]
        bot_coor = coor[:int(conn[weak[-1], 1] + 1), :]
        top_coor = coor[conn[weak[0], 3]:, :]
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
        assert np.allclose(self.m_coor[self.m_nodesLeftEdge, 1], self.m_coor[self.m_nodesRightEdge, 1])
        assert np.allclose(self.m_coor[self.m_nodesBottomEdge, 0], self.m_coor[self.m_nodesTopEdge, 0])
        assert self.m_nodesBottomEdge.size == Nx + 1
        assert self.m_nodesTopEdge.size == Nx + 1

# ==================================================================================================

class TopLayerElastic(MyMesh):

    def __init__(self, Nx, Ny, h):

        mesh = GooseFEM.Mesh.Quad4.FineLayer(Nx, Ny, h)

        coor = mesh.coor()
        conn = mesh.conn()
        weak = mesh.elementsMiddleLayer()

        top_conn = conn[int(weak[-1] + 1):, :] - conn[weak[0], 3]
        top_coor = coor[conn[weak[0], 3]:, :]
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
        bot_conn = conn[:weak[0], :]
        bot_coor = coor[:int(conn[weak[-1], 1] + 1), :]
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

def generate(myversion, filename, nplates, seed, rid):

    N = 3 ** 6
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

    stitch.push_back(layer_bot.coor(), layer_bot.conn(), layer_bot.nodesBottomEdge(), layer_bot.nodesTopEdge())
    stitch.push_back(layer_plas.coor(), layer_plas.conn(), layer_plas.nodesBottomEdge(), layer_plas.nodesTopEdge())
    left += [layer_bot.nodesLeftOpenEdge(), layer_plas.nodesLeftEdge()]
    right += [layer_bot.nodesRightOpenEdge(), layer_plas.nodesRightEdge()]

    if nplates > 2:
        for i in range(nplates - 2):
            stitch.push_back(layer_elas.coor(), layer_elas.conn(), layer_elas.nodesBottomEdge(), layer_elas.nodesTopEdge())
            stitch.push_back(layer_plas.coor(), layer_plas.conn(), layer_plas.nodesBottomEdge(), layer_plas.nodesTopEdge())
            left += [layer_elas.nodesLeftOpenEdge(), layer_plas.nodesLeftEdge()]
            right += [layer_elas.nodesRightOpenEdge(), layer_plas.nodesRightEdge()]

    stitch.push_back(layer_top.coor(), layer_top.conn(), layer_top.nodesBottomEdge(), layer_top.nodesTopEdge())
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
    H = np.max(coor[:, 1]) - np.min(coor[:, 1])
    Hi = [(coor[conn[stitch.elemmap(i)[-1], 3], 1] + coor[conn[stitch.elemmap(i)[0], 0], 1]) / 2.0 for i in range(nlayer)]

    dofs = stitch.dofs()
    dofs[right, :] = dofs[left, :]
    dofs[top[-1], :] = dofs[top[0], :]
    dofs[bottom[-1], :] = dofs[bottom[0], :]
    dofs = GooseFEM.Mesh.renumber(dofs)

    iip = np.concatenate((dofs[bottom[:-1], :].ravel(), dofs[top[:-1], :].ravel()))

    elastic = []
    plastic = []

    for i, ispl in enumerate(is_plastic):
        if ispl:
            plastic += list(stitch.elemmap(i))
        else:
            elastic += list(stitch.elemmap(i))

    initstate = seed + np.arange(N * (nplates - 1)).astype(np.int64)
    initseq = np.zeros_like(initstate)
    generators = prrng.pcg32_array(initstate, initseq)

    k = 2.0
    eps0 = 0.5 * 1e-4
    eps_offset = 1e-2 * (2.0 * eps0)
    nchunk = 6000

    # epsy = eps_offset + (2.0 * eps0) * generators.weibull([nchunk], k)
    # epsy[0: left, 0] *= init_factor
    # epsy[right: N, 0] *= init_factor
    # epsy = np.cumsum(epsy, axis=1)

    c = 1.0
    G = 1.0
    K = 4.5 * G # consistent with PMMA
    rho = G / c ** 2.0
    qL = 2.0 * np.pi / L
    qh = 2.0 * np.pi / h
    alpha = np.sqrt(2.0) * qL * c * rho

    dt = (1.0 / (c * qh)) / 10.0

    with h5py.File(filename, "w") as data:

        key = "/coor"
        data[key] = coor
        data[key].attrs["desc"] = "Nodal coordinates [nnode, ndim]"

        key = "/conn"
        data[key] = conn
        data[key].attrs["desc"] = "Connectivity (Quad4: nne = 4) [nelem, nne]"

        key = "/dofs"
        data[key] = dofs
        data[key].attrs["desc"] = "DOFs per node, accounting for semi-periodicity [nnode, ndim]"

        key = "/iip"
        data[key] = iip
        data[key].attrs["desc"] = "Prescribed DOFs [nnp]"

        key = "/run/epsd/kick"
        data[key] = eps0 * 1e-4
        data[key].attrs["desc"] = "Strain kick to apply"

        key = "/run/dt"
        data[key] = dt
        data[key].attrs["desc"] = "Time step"

        key = "/rho"
        data[key] = rho * np.ones((nelem))
        data[key].attrs["desc"] = "Mass density [nelem]"

        key = "/damping/alpha"
        data[key] = alpha * np.ones((nelem))
        data[key].attrs["desc"] = "Damping coefficient (density) [nelem]"

        key = "/cusp/elem"
        data[key] = plastic
        data[key].attrs["desc"] = "Plastic elements with cusp potential [nplastic]"

        key = "/cusp/K"
        data[key] = K * np.ones((len(plastic)))
        data[key].attrs["desc"] = "Bulk modulus for elements in '/cusp/elem' [nplastic]"

        key = "/cusp/G"
        data[key] = G * np.ones((len(plastic)))
        data[key].attrs["desc"] = "Shear modulus for elements in '/cusp/elem' [nplastic]"

        key = "/cusp/epsy/initstate"
        data[key] = initstate
        data[key].attrs["desc"] = "State to use to initialise prrng::pcg32"

        key = "/cusp/epsy/initseq"
        data[key] = initseq
        data[key].attrs["desc"] = "Sequence to use to initialise prrng::pcg32"

        key = "/cusp/epsy/k"
        data[key] = k
        data[key].attrs["desc"] = "Shape factor of Weibull distribution"

        key = "/cusp/epsy/eps0"
        data[key] = eps0
        data[key].attrs["desc"] = "Yield strain normalisation: multiply all yield strains with twice this factor"

        key = "/cusp/epsy/eps_offset"
        data[key] = eps_offset
        data[key].attrs["desc"] = "Yield strain offset: add this (small) offset to each yield strain, after normalisation!"

        key = "/cusp/epsy/nchunk"
        data[key] = nchunk
        data[key].attrs["desc"] = "Chunk size"

        key = "/elastic/elem"
        data[key] = elastic
        data[key].attrs["desc"] = "Elastic elements [nelem - N]"

        key = "/elastic/K"
        data[key] = K * np.ones((len(elastic)))
        data[key].attrs["desc"] = "Bulk modulus for elements in '/elastic/elem' [nelem - N]"

        key = "/elastic/G"
        data[key] = G * np.ones((len(elastic)))
        data[key].attrs["desc"] = "Shear modulus for elements in '/elastic/elem' [nelem - N]"

        key = "/meta/normalisation/N"
        data[key] = N
        data[key].attrs["desc"] = "Number of blocks along each plastic layer"

        key = "/meta/normalisation/l"
        data[key] = h
        data[key].attrs["desc"] = "Elementary block size"

        key = "/meta/normalisation/rho"
        data[key] = rho
        data[key].attrs["desc"] = "Elementary density"

        key = "/meta/normalisation/G"
        data[key] = G
        data[key].attrs["desc"] = "Uniform shear modulus == 2 mu"

        key = "/meta/normalisation/K"
        data[key] = K
        data[key].attrs["desc"] = "Uniform bulk modulus == kappa"

        key = "/meta/normalisation/eps"
        data[key] = eps0
        data[key].attrs["desc"] = "Typical yield strain"

        key = "/meta/normalisation/sig"
        data[key] = 2.0 * G * eps0
        data[key].attrs["desc"] = "== 2 G eps0"

        key = "/meta/seed_base"
        data[key] = seed
        data[key].attrs["desc"] = "Basic seed == 'unique' identifier"

        key = f"/meta/{basename}/{genscript}"
        data[key] = myversion
        data[key].attrs["desc"] = "Version when generating"

        elemmap = stitch.elemmap()
        nodemap = stitch.nodemap()

        key = "/layers/stored"
        data[key] = np.arange(len(elemmap)).astype(np.int64)
        data[key].attrs["desc"] = "Layers in simulation"

        for i in range(len(elemmap)):
            data["/layers/{0:d}/elemmap".format(i)] = elemmap[i]
            data["/layers/{0:d}/nodemap".format(i)] = nodemap[i]

        key = "/layers/is_plastic"
        data[key] = is_plastic
        data[key].attrs["desc"] = "Per layer: true is the layer is plastic"

# ----------

N = 3 ** 6
seed = 0
max_plates = 100

for rid, nplates in itertools.product(range(3), [2, 3, 4, 5]):
    generate(myversion, "id={0:d}_nplates={1:d}.h5".format(rid, nplates), nplates, seed, rid)
    seed += N * (max_plates - 1)

