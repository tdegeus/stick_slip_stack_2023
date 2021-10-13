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

        historic = os.path.splitext(__file__)[0] + ".h5"
        dirname = "mytest"
        idname = "id=0.h5"
        filename = os.path.join(dirname, idname)
        checkname = os.path.join(dirname, my.FixedLever.file_defaults["cli_find_completed"])
        delta_gamma = np.concatenate(
            (
                np.zeros(1, dtype=float),
                1e-4 * np.ones(4, dtype=float),
                1e-5 * np.ones(100, dtype=float),
            )
        )

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

        with h5py.File(filename, "r") as file:
            system = my.FixedLever.init(file)
            out = my.FixedLever.basic_output(system, file, verbose=False)

        with h5py.File(historic, "r") as file:
            self.assertTrue(np.allclose(file["epsd"][...], out["epsd"]))
            self.assertTrue(np.allclose(file["sigd"][...], out["sigd"]))
            self.assertTrue(np.allclose(file["drive_fx"][...], out["drive_fx"]))

        shutil.rmtree(dirname)


if __name__ == "__main__":

    unittest.main()
