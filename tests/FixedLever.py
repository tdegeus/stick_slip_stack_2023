import os
import shutil
import sys
import unittest

import GooseHDF5 as g5
import h5py
import numpy as np

root = os.path.join(os.path.dirname(__file__), "..")
if os.path.exists(os.path.join(root, "mycode_lever", "_version.py")):
    sys.path.insert(0, os.path.abspath(root))

import mycode_lever as my  # noqa: E402


class MyTests(unittest.TestCase):
    """
    Tests
    """

    def test_generate(self):

        dirname = "mytest"
        file_a = os.path.join(dirname, "id=0.h5")

        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        if os.path.exists(file_a):
            os.remove(file_a)

        my.FixedLever.generate(
            filename=file_a,
            N=9,
            nplates=2,
            seed=0,
            k_drive=1e-3,
            symmetric=True,
        )

        my.FixedLever.cli_generate(["--develop", "-N", 9, "-n", 1, dirname])

        file_b = os.path.join(dirname, "id=000_nplates=2_kplate=1e-03_symmetric=1.h5")
        self.assertTrue(os.path.isfile(file_b))

        with h5py.File(file_a, "r") as source:
            with h5py.File(file_b, "r") as dest:
                for path in g5.getdatasets(source):
                    self.assertTrue(g5.equal(source, dest, path))

        shutil.rmtree(dirname)

    def test_compare(self):

        dirname = "mytest"
        file_a = os.path.join(dirname, "id=0.h5")
        file_b = os.path.join(dirname, "id=1.h5")

        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        for file in [file_a, file_b]:
            if os.path.exists(file):
                os.remove(file)

        my.FixedLever.generate(
            filename=file_a,
            N=9,
            nplates=2,
            seed=0,
            k_drive=1e-3,
            symmetric=True,
        )

        my.FixedLever.generate(
            filename=file_b,
            N=9,
            nplates=2,
            seed=0,
            k_drive=2e-3,
            symmetric=True,
        )

        ret = my.System.cli_compare([file_a, file_b])
        expect = {"!=": ["/drive/delta_gamma", "/drive/k"]}
        self.assertEqual(ret, expect)

        shutil.rmtree(dirname)

    def test_small(self):

        dirname = "mytest"
        idname = "id=0.h5"
        filename = os.path.join(dirname, idname)
        checkname = os.path.join(dirname, my.FixedLever.file_defaults["cli_find_completed"])
        infoname = os.path.join(dirname, my.FixedLever.file_defaults["cli_ensembleinfo"])
        eventname = os.path.join(dirname, my.FixedLever.file_defaults["cli_rerun_event"])
        delta_gamma = 1e-4 * np.ones(5)
        delta_gamma[0] = 0

        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        my.FixedLever.generate(
            filename=filename,
            N=9,
            nplates=2,
            seed=0,
            k_drive=1e-3,
            symmetric=True,
            delta_gamma=delta_gamma,
        )

        my.FixedLever.cli_run([filename, "--develop"])
        completed = my.FixedLever.cli_find_completed(["-f", "-o", checkname, filename])
        self.assertEqual(completed, [filename])

        my.FixedLever.cli_ensembleinfo(["-o", infoname, filename])
        my.FixedLever.cli_view_paraview([filename])
        my.FixedLever.cli_rerun_event([filename, "-i", 1, "-o", eventname])
        my.FixedLever.cli_job_rerun_multislip([infoname, "-o", dirname])

        shutil.rmtree(dirname)


if __name__ == "__main__":

    unittest.main()
