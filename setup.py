import os

from setuptools import setup

from flavor.version import __version__

os.environ["PYTHONWARNINGS"] = "ignore"

setup(
    version=__version__,
    scripts=[
        "bin/flavor-fl",
        "bin/flavor-agg",
        "bin/flavor-fv",
        "bin/check-fl",
        "bin/check-agg",
        "bin/check-fv",
    ],
)
