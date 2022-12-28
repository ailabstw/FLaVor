from setuptools import find_packages, setup

import flavors


def parse_requirements(filename):
    with open(filename, "r") as f:
        reqs = f.read()
    return reqs.split("\n")


setup(
    name="flavors",
    version=flavors.__version__,
    packages=find_packages(exclude=["dockerfile*", "example*"]),
    install_requires=parse_requirements("requirements.txt"),
)
