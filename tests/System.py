import os
import sys
import unittest

import numpy as np

root = os.path.join(os.path.dirname(__file__), "..")
if os.path.exists(os.path.join(root, "mycode_lever", "_version.py")):
    sys.path.insert(0, os.path.abspath(root))

import mycode_lever as my  # noqa: E402


class MyTests(unittest.TestCase):
    """
    Tests
    """

    def test_indices(self):

        indices = np.array([0, -1, 1, -2, 2, -3, 3, -4, 4])

        for q in range(0, my.System.nlayer(), 2):
            self.assertEqual(my.System.layer2plate(q), indices[q])

        for q in range(1, my.System.nlayer(), 2):
            self.assertEqual(my.System.layer2interface(q), -indices[q])

        for i in range(0, my.System.nplate() + 1):
            self.assertEqual(my.System.plate2layer(i), np.argwhere(indices == i).ravel()[0])

        for i in range(1, my.System.nplate() + 1):
            self.assertEqual(my.System.interface2layer(i), np.argwhere(indices == -i).ravel()[0])

        self.assertEqual(indices[-1], my.System.nplate())


if __name__ == "__main__":

    unittest.main()
