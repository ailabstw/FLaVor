import os

from setuptools import find_packages, setup

import flavors

os.environ["PYTHONWARNINGS"] = "ignore"


def parse_requirements(filename):
    with open(filename, "r") as f:
        reqs = f.read()
    return reqs.split("\n")


setup(
    name="flavors",
    version=flavors.__version__,
    packages=find_packages(exclude=["dockerfile*", "example*", "script*"]),
    install_requires=parse_requirements("requirements.txt"),
    scripts=[
        "bin/flavors-fv",
        "bin/flavors-fl",
        "bin/flavors-check",
    ],
)
