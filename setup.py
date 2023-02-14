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
    description="AILabs Federated Learning and Validation Framework",
    author="Keng-Chi Liu",
    author_email="calvin89029@gmail.com",
    packages=find_packages(exclude=["dockerfile*", "example*", "script*"]),
    install_requires=parse_requirements("requirements.txt"),
    data_files=[
        ("schema", ["schema/FLresult.json", "schema/FVresult.json"]),
    ],
    scripts=[
        "bin/flavor-fl",
        "bin/flavor-agg",
        "bin/flavor-fv",
        "bin/check-fl",
        "bin/check-agg",
        "bin/check-fv",
    ],
)
