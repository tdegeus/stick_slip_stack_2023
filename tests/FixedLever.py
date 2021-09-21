import os
import sys
import unittest

root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(root))
import mycode_lever as my  # noqa: E402


class MyTests(unittest.TestCase):
    def test_run(self):

        my.FixedLever.generate(
            filename="mytest.h5",
            N=9,
            nplates=2,
            seed=0,
            k_drive=1e-3,
            symmetric=True,
            delta_gamma=[0, 0.005, 0.01],
        )


if __name__ == "__main__":

    unittest.main()
