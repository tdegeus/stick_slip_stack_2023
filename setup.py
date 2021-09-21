from setuptools import find_packages
from setuptools import setup

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
        "console_scripts": [
            "FixedLever = mycode_lever.FixedLever:cli_run",
            "FixedLever_EnsembleInfo = mycode_lever.FixedLever:cli_ensembleinfo",
            "FixedLever_Events = mycode_lever.FixedLever:cli_rerun_event",
            "FixedLever_EventsJob = mycode_lever.FixedLever:cli_job_rerun_multislip",
        ]
    },
)
