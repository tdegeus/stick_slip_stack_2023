import re

from setuptools import find_packages
from setuptools import setup


def read_entry_points(module):

    entry_points = []

    with open(f"mycode_lever/{module}.py") as file:
        contents = file.read()
        eps = contents.split("entry_points = dict(\n")[1].split(")\n")[0].split("\n")
        eps = list(filter(None, eps))
        for ep in eps:
            regex = r"([\ ]*)(\w*)([\ ]*\=[\ ]*)(\")(\w*)(\".*)"
            _, _, func, _, _, name, _, _ = re.split(regex, ep)
            entry_points += [f"{name} = mycode_lever.{module}:{func}"]

    return entry_points


entry_points = []
entry_points += read_entry_points("FixedLever")
entry_points += read_entry_points("slurm")
entry_points += read_entry_points("System")


setup(
    name="mycode_lever",
    license="MIT",
    author="Tom de Geus",
    author_email="tom@geus.me",
    description="Code for running simulations driven by a lever",
    packages=find_packages(),
    use_scm_version={"write_to": "mycode_lever/_version.py"},
    setup_requires=["setuptools_scm"],
    entry_points={"console_scripts": entry_points},
)
