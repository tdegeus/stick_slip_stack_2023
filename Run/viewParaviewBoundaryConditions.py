import argparse
import os
import tqdm
import h5py
import numpy as np
import prrng
import GooseFEM as gf
import GMatElastoPlasticQPot.Cartesian2d as gmat
import FrictionQPotFEM.UniformMultiLayerIndividualDrive2d as model
import XDMFWrite_h5py as xh

parser = argparse.ArgumentParser()
parser.add_argument("files", type=str, nargs="*")
parser.add_argument("-p", "--prefix", type=str, default="paraview")
args = parser.parse_args()
assert np.all([os.path.isfile(file) for file in args.files])
assert len(args.files) > 0

def read_epsy(data, N):

    initstate = data["/cusp/epsy/initstate"][...]
    initseq = data["/cusp/epsy/initseq"][...]
    eps_offset = data["/cusp/epsy/eps_offset"][...]
    eps0 = data["/cusp/epsy/eps0"][...]
    k = data["/cusp/epsy/k"][...]
    nchunk = data["/cusp/epsy/nchunk"][...]

    generators = prrng.pcg32_array(initstate, initseq)

    epsy = generators.weibull([nchunk], k)
    epsy *= (2.0 * eps0)
    epsy += eps_offset
    epsy = np.cumsum(epsy, 1)

    return epsy

for file in tqdm.tqdm(args.files):

    outbasename = "{0:s}_{1:s}".format(args.prefix, os.path.splitext(file)[0])

    with h5py.File(outbasename + ".hdf5", "w") as output:

        with h5py.File(file, "r") as data:

            layers = data["/layers/stored"][...]
            elemmap = []
            nodemap = []

            for layer in layers:
                elemmap += [data["/layers/{0:d}/elemmap".format(layer)][...]]
                nodemap += [data["/layers/{0:d}/nodemap".format(layer)][...]]

            system = model.System(
                data["/coor"][...],
                data["/conn"][...],
                data["/dofs"][...],
                data["/iip"][...],
                elemmap,
                nodemap,
                data["/layers/is_plastic"][...])

            system.setDriveStiffness(data["/drive/k"][...], data["/drive/symmetric"][...])
            system.setMassMatrix(data["/rho"][...])
            system.setDampingMatrix(data["/damping/alpha"][...])

            system.setElastic(
                data["/elastic/K"][...],
                data["/elastic/G"][...])

            system.setPlastic(
                data["/cusp/K"][...],
                data["/cusp/G"][...],
                read_epsy(data, system.plastic().size))

            system.setDt(data["/run/dt"][...])

            vector = system.vector()

            # mesh

            output["/coor"] = system.coor()
            output["/conn"] = system.conn()

            fields = xh.Unstructured(output, "/coor", "/conn", "Quadrilateral")

            # prescribed DOFs (as nodevec)

            prescribed = np.zeros(vector.shape_dofval())
            prescribed[data["/iip"][...]] = 1
            prescribed = vector.AsNode(prescribed)

            key = "/prescribed"
            output["/prescribed"] = xh.as3d(prescribed)
            fields += xh.Attribute(output, "/prescribed", "Node")

            # periodic DOFs (as nodevec)

            periodic = np.random.random(vector.shape_nodevec())
            test = vector.AsNode(vector.AssembleDofs(periodic))
            periodic = periodic != test

            key = "/periodic"
            output["/periodic"] = xh.as3d(periodic)
            fields += xh.Attribute(output, "/periodic", "Node")

            # nodes per layer

            for i in range(system.nlayers()):

                nodevec = np.zeros(vector.shape_nodevec())
                nodevec[system.layerNodes(i), :] = 1

                key = "/layer/{0:d}/nodes".format(i)
                output[key] = xh.as3d(nodevec)
                fields += xh.Attribute(output, key, "Node")

            # elements per layer

            for i in range(system.nlayers()):

                el = np.zeros((system.conn().shape[0]))
                el[system.layerElements(i)] = 1

                key = "/layer/{0:d}/elements".format(i)
                output[key] = el
                fields += xh.Attribute(output, key, "Cell")

            # is plastic

            isplas0 = np.zeros((system.conn().shape[0]))
            isplas1 = np.mean(system.material().isPlastic(), axis=1)

            for i in range(system.nlayers()):

                if system.layerIsPlastic(i):
                    isplas0[system.layerElements(i)] += 1

            key = "/is_plastic/0"
            output[key] = isplas0
            fields += xh.Attribute(output, key, "Cell")

            key = "/is_plastic/1"
            output[key] = isplas1
            fields += xh.Attribute(output, key, "Cell")

            # write output

            grid = xh.Grid(fields)
            xh.write(grid, outbasename + ".xdmf")



