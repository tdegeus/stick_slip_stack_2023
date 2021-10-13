import GooseFEM
import numpy as np


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
