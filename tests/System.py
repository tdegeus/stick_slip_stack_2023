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

    def test_SequenceDelta(self):

        delta = 0.5 + np.random.random(10)
        delta_inf = np.concatenate((delta, delta[-1] * np.ones(30)))
        cumsum_inf = np.cumsum(delta_inf)

        d = my.System.SequenceDelta(delta)

        for i in range(cumsum_inf.size):
            self.assertTrue(np.isclose(d.get_delta(i), delta_inf[i]))
            self.assertTrue(np.isclose(d.get_cumsum(i), cumsum_inf[i]))

        self.assertTrue(np.allclose(d.list_delta(delta_inf.size), delta_inf))
        self.assertTrue(np.allclose(d.list_delta(cumsum_inf.size), cumsum_inf))


if __name__ == "__main__":

    unittest.main()
