import unittest
import os
import sys

root = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(os.path.abspath(root))
import mycode_lever as my  # noqa: E402


class MyTests(unittest.TestCase):

    def test_has_uncomitted(self):

        self.assertTrue(my.tag.has_uncommited("4.4.dev1+g423e6a8.d20210902"))
        self.assertFalse(my.tag.has_uncommited("4.4.dev1+g423e6a8"))
        self.assertFalse(my.tag.has_uncommited("4.4.dev1"))
        self.assertFalse(my.tag.has_uncommited("4.4"))

    def test_any_has_uncommitted(self):

        self.assertTrue(my.tag.any_has_uncommitted(["main=3.2.1", "other=4.4.dev1+g423e6a8.d20210902"]))
        self.assertFalse(my.tag.any_has_uncommitted(["main=3.2.1", "other=4.4.dev1+g423e6a8"]))
        self.assertFalse(my.tag.any_has_uncommitted(["main=3.2.1", "other=4.4.dev1"]))
        self.assertFalse(my.tag.any_has_uncommitted(["main=3.2.1", "other=4.4"]))

    def test_greater_equal(self):

        self.assertFalse(my.tag.greater_equal("4.4.dev1+g423e6a8.d20210902", "4.4"))
        self.assertFalse(my.tag.greater_equal("4.4.dev1+g423e6a8", "4.4"))
        self.assertFalse(my.tag.greater_equal("4.4.dev1", "4.4"))
        self.assertTrue(my.tag.greater_equal("4.4", "4.4"))


    def test_all_greater_equal(self):

        a = ["main=3.2.1", "other=4.4"]
        b = ["main=3.2.0", "other=4.4", "more=3.0.0"]
        self.assertTrue(my.tag.all_greater_equal(a, b))

        a = ["main=3.2.1", "other=4.4"]
        b = ["main=3.2.1.dev1", "other=4.4", "more=3.0.0"]
        self.assertTrue(my.tag.all_greater_equal(a, b))

        a = ["main=3.2.1", "other=4.4"]
        b = ["main=3.2.1.dev1+g423e6a8", "other=4.4", "more=3.0.0"]
        self.assertTrue(my.tag.all_greater_equal(a, b))

        a = ["main=3.2.1", "other=4.4"]
        b = ["main=3.2.1.dev1+g423e6a8.d20210902", "other=4.4", "more=3.0.0"]
        self.assertTrue(my.tag.all_greater_equal(a, b))

if __name__ == "__main__":

    unittest.main()
