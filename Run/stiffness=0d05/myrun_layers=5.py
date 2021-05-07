import h5py
import numpy as np
import GooseFEM
import uuid
import FrictionQPotFEM.UniformMultiLayerIndividualDrive2d as model

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

        extra = GooseFEM.Mesh.Quad4.Regular(int(np.ceil(Nx * h / H)), int(np.ceil((Ny * h - Ly) / H)), H)

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

        extra = GooseFEM.Mesh.Quad4.Regular(int(np.ceil(Nx * h / H)), int(np.ceil((Ny * h - Ly) / H)), H)

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

N = 3 ** 5
M = N
h = np.pi
nlayer = 9
stacking = [False, True, False, True, False, True, False, True, False]

layer_bot = BottomLayerElastic(N, M, h)
layer_top = TopLayerElastic(N, M, h)
layer_elas = LayerElastic(N, M, h)
layer_plas = GooseFEM.Mesh.Quad4.Regular(N, 1, h)

stitch = GooseFEM.Mesh.Vstack()

stitch.push_back(layer_bot.coor(), layer_bot.conn(), layer_bot.nodesBottomEdge(), layer_bot.nodesTopEdge())
stitch.push_back(layer_plas.coor(), layer_plas.conn(), layer_plas.nodesBottomEdge(), layer_plas.nodesTopEdge())
stitch.push_back(layer_elas.coor(), layer_elas.conn(), layer_elas.nodesBottomEdge(), layer_elas.nodesTopEdge())
stitch.push_back(layer_plas.coor(), layer_plas.conn(), layer_plas.nodesBottomEdge(), layer_plas.nodesTopEdge())
stitch.push_back(layer_elas.coor(), layer_elas.conn(), layer_elas.nodesBottomEdge(), layer_elas.nodesTopEdge())
stitch.push_back(layer_plas.coor(), layer_plas.conn(), layer_plas.nodesBottomEdge(), layer_plas.nodesTopEdge())
stitch.push_back(layer_elas.coor(), layer_elas.conn(), layer_elas.nodesBottomEdge(), layer_elas.nodesTopEdge())
stitch.push_back(layer_plas.coor(), layer_plas.conn(), layer_plas.nodesBottomEdge(), layer_plas.nodesTopEdge())
stitch.push_back(layer_top.coor(), layer_top.conn(), layer_top.nodesBottomEdge(), layer_top.nodesTopEdge())

left = stitch.nodeset([
    layer_bot.nodesLeftOpenEdge(),
    layer_plas.nodesLeftEdge(),
    layer_elas.nodesLeftOpenEdge(),
    layer_plas.nodesLeftEdge(),
    layer_elas.nodesLeftOpenEdge(),
    layer_plas.nodesLeftEdge(),
    layer_elas.nodesLeftOpenEdge(),
    layer_plas.nodesLeftEdge(),
    layer_top.nodesLeftOpenEdge(),
])

right = stitch.nodeset([
    layer_bot.nodesRightOpenEdge(),
    layer_plas.nodesRightEdge(),
    layer_elas.nodesRightOpenEdge(),
    layer_plas.nodesRightEdge(),
    layer_elas.nodesRightOpenEdge(),
    layer_plas.nodesRightEdge(),
    layer_elas.nodesRightOpenEdge(),
    layer_plas.nodesRightEdge(),
    layer_top.nodesRightOpenEdge(),
])

bottom = stitch.nodeset(layer_bot.nodesBottomEdge(), 0)
top = stitch.nodeset(layer_top.nodesTopEdge(), 2)

nelem = stitch.nelem()
coor = stitch.coor()
conn = stitch.conn()

L = np.max(coor[:, 0]) - np.min(coor[:, 0])
H = np.max(coor[:, 1]) - np.min(coor[:, 1])
Hi = [(coor[conn[stitch.elemmap(i)[-1], 3], 1] + coor[conn[stitch.elemmap(i)[0], 0], 1]) / 2.0 for i in range(nlayer)]

dofs = stitch.dofs()
dofs[right, :] = dofs[left, :]
dofs[top[0], 0] = dofs[top[-1], 0]
dofs[bottom[0], 0] = dofs[bottom[-1], 0]
dofs = GooseFEM.Mesh.renumber(dofs)

iip = np.concatenate((dofs[bottom, :].ravel(), dofs[top, 1].ravel()))

elastic = []
plastic = []

for i, isplastic in enumerate(stacking):
    if isplastic:
        plastic += list(stitch.elemmap(i))
    else:
        elastic += list(stitch.elemmap(i))

k = 2.0
realization = str(uuid.uuid4())
epsy = 1e-5 + 1e-3 * np.random.weibull(k, size=1000 * len(plastic)).reshape(len(plastic), -1)
epsy[:,0] = 1e-5 + 1e-3 * np.random.random(len(plastic))
epsy = np.cumsum(epsy, axis=1)
i = np.min(np.where(np.min(epsy, axis=0) > 0.55)[0])
epsy = epsy[:, :i]

c = 1.0
G = 1.0
K = 10.0 * G
rho = G / c ** 2.0
qL = 2.0 * np.pi / L
qh = 2.0 * np.pi / h
alpha = np.sqrt(2.0) * qL * c * rho

dt = (1.0 / (c * qh)) / 10.0

with h5py.File("myrun_layers=5.h5", "w") as data:
    data["/epsy"] = epsy

system = model.System(coor, conn, dofs, iip, stitch.elemmap(), stitch.nodemap(), stacking)
system.setMassMatrix(rho * np.ones((nelem)))
system.setDampingMatrix(alpha * np.ones((nelem)))
system.setElastic(K * np.ones((len(elastic))), G * np.ones((len(elastic))))
system.setPlastic(K * np.ones((len(plastic))), G * np.ones((len(plastic))), epsy)
system.setDt(dt)
system.setDriveStiffness(0.05)

ubar = np.zeros((nlayer, 2))
drive = np.zeros((nlayer, 2), dtype=bool)
for i, isplastic in enumerate(stacking):
    if not isplastic and i > 0:
        drive[i, 0] = True

system.layerSetUbar(ubar, drive)

for inc in range(1000):

    for i, isplastic in enumerate(stacking):
        if not isplastic and i > 0:
            ubar[i, 0] = float(inc) * 0.01 * 1e-3 * Hi[i]

    system.layerSetUbar(ubar, drive)

    print(system.minimise())

    with h5py.File("myrun_layers=5.h5", "a") as data:
        data["/disp/{0:d}".format(inc)] = system.u()



