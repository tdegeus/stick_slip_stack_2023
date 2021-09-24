import os
import shutil
import sys
import unittest
import GooseHDF5 as g5
import h5py

root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(root))
import mycode_lever as my  # noqa: E402


class MyTests(unittest.TestCase):

    def test_generate(self):

        dirname = "mytest"
        file_a = os.path.join(dirname, "id=0.h5")

        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        my.FixedLever.generate(
            filename=file_a,
            N=9,
            nplates=2,
            seed=0,
            k_drive=1e-3,
            symmetric=True,
        )

        my.FixedLever.cli_generate(["-N", 9, "-n", 1, dirname])

        file_b = os.path.join(dirname, "id=000_nplates=2_kplate=1e-03_symmetric=1.h5")
        self.assertTrue(os.path.isfile(file_b))

        with h5py.File(file_a, "r") as source:
            with h5py.File(file_b, "r") as dest:
                for path in g5.getdatasets(source):
                    self.assertTrue(g5.equal(source, dest, path))

        shutil.rmtree(dirname)

    def test_small(self):

        dirname = "mytest"
        idname = "id=0.h5"
        filename = os.path.join(dirname, idname)
        infoname = os.path.join(dirname, "EnsembleInfo.h5")
        eventname = os.path.join(dirname, "events.h5")

        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        my.FixedLever.generate(
            filename=filename,
            N=9,
            nplates=2,
            seed=0,
            k_drive=1e-3,
            symmetric=True,
            delta_gamma=[0, 0.005, 0.01, 0.015],
        )

        my.FixedLever.cli_run([filename, "-f"])
        my.FixedLever.cli_ensembleinfo(["-o", infoname, filename])
        my.FixedLever.cli_view_paraview([filename])
        my.FixedLever.cli_rerun_event([filename, "-i", 1, "-o", eventname])
        my.FixedLever.cli_job_rerun_multislip([infoname, "-o", dirname])

        shutil.rmtree(dirname)


if __name__ == "__main__":

    unittest.main()
