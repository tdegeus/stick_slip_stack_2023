import os
import shutil
import sys
import unittest

root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(root))
import mycode_lever as my  # noqa: E402


class MyTests(unittest.TestCase):
    def test_exec_cmd(self):

        cmd = 'echo "hello world"'
        script = my.slurm.script_exec(cmd)
        self.assertEqual(script.split("\n")[-1], "stdbuf -o0 -e0 " + cmd)

        _ = my.slurm.script_exec(cmd, conda=dict(condabase="my"))
        _ = my.slurm.script_exec(cmd, conda=("my", "/root"))
        _ = my.slurm.script_exec(cmd, conda=None)
        _ = my.slurm.script_exec(cmd, conda=False)

    def test_cli_serial_group(self):

        dirname = "mytest"
        filename = os.path.join(dirname, "foo.txt")

        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        with open(filename, "w") as file:
            file.write("")

        my.slurm.cli_serial_group(["-o", dirname, "-c", "dummy", filename])

        shutil.rmtree(dirname)


if __name__ == "__main__":

    unittest.main()
