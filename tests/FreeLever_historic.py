import os
import shutil
import sys
import unittest

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

    def test_small(self):
        """
        Generate and run a small sample.
        """

        historic = os.path.splitext(__file__)[0] + ".h5"
        dirname = "mytest"
        idname = "id=0.h5"
        filename = os.path.join(dirname, idname)

        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        my.FreeLever.generate(
            filename=filename,
            N=9,
            nplates=2,
            seed=0,
            k_drive=1e-3,
            symmetric=True,
            test_mode=True,
        )

        my.FreeLever.cli_run([filename, "--develop"])

        with h5py.File(filename, "r") as file:
            system = my.FreeLever.init(file)
            out = my.FreeLever.basic_output(system, file, verbose=False)

        # with h5py.File(historic, "w") as file:
        #     file["epsd"] = out["epsd"]
        #     file["sigd"] = out["sigd"]
        #     file["fxdrive_layers"] = out["fxdrive_layers"]

        with h5py.File(historic, "r") as file:
            self.assertTrue(np.allclose(file["epsd"][...], out["epsd"]))
            self.assertTrue(np.allclose(file["sigd"][...], out["sigd"]))
            self.assertTrue(np.allclose(file["fxdrive_layers"][...], out["fxdrive_layers"]))

        shutil.rmtree(dirname)


if __name__ == "__main__":

    unittest.main()
