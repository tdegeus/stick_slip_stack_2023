import re

from setuptools import find_packages
from setuptools import setup

entry_points = []

with open("mycode_lever/FixedLever.py", "r") as file:
    contents = file.read()
    eps = list(filter(None, contents.split("entry_points = dict(\n")[1].split(")\n")[0].split("\n")))
    for ep in eps:
        _, _, func, _, _, name, _, _ = re.split(r"([\ ]*)(\w*)([\ ]*\=[\ ]*)(\")(\w*)(\".*)", ep)
        entry_points += [f"{name} = mycode_lever.FixedLever:{func}"]


setup(
    name="mycode_lever",
    license="MIT",
    author="Tom de Geus",
    author_email="tom@geus.me",
    description="Code for running simulations driven by a lever",
    packages=find_packages(),
    use_scm_version={"write_to": "mycode_lever/_version.py"},
    setup_requires=["setuptools_scm"],
    entry_points={
        "console_scripts": entry_points
    },
)
