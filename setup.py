#!/usr/bin/env python

import os
from setuptools import setup


setup(
    name="awkward_extras",
    version="0.0.1",
    description="Extras for Awkward1",
    url="",
    maintainer="Martin Durant",
    maintainer_email="mdurant@anaconda.com",
    license="BSD",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    keywords=["arrays"],
    packages=["awkward_pandas", "awkward_zarr"],
    install_requires=["awkward1"],
    long_description=(
        open("README.rst").read() if os.path.exists("README.rst") else ""
    ),
    extras_require={"pandas": ["pandas"],
                    "zarr": ["zarr"],
                    "complete": ["pandas", "zarr"]},
    python_requires=">=3.6",
    zip_safe=False,
)
