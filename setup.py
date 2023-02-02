import os

from setuptools import find_packages, setup

import flavor

os.environ["PYTHONWARNINGS"] = "ignore"


def parse_requirements(filename):
    with open(filename, "r") as f:
        reqs = f.read()
    return reqs.split("\n")


setup(
    name="flavor",
    version=flavor.__version__,
    packages=find_packages(exclude=["dockerfile*", "example*", "script*"]),
    install_requires=parse_requirements("requirements.txt"),
    scripts=[
        "bin/flavor-fl",
        "bin/flavor-agg",
        "bin/flavor-fv",
        "bin/check-fl",
        "bin/check-fv",
    ],
)
