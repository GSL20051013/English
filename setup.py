# SPDX-License-Identifier: LicenseRef-GSL20051013-english-noncommercial
# Copyright (c) 2024 GSL20051013
# See LICENSE for full terms. Commercial use requires a paid license.
"""Build script for the optional Cython extension.

Usage
-----
To compile the Cython extension in-place::

    python setup.py build_ext --inplace

The compiled module (``english_core*.so`` / ``english_core*.pyd``) is placed
inside the ``Geemeth/`` package directory.  When present, ``english.py``
imports the C-compiled hot-paths from it automatically, falling back to the
pure-Python implementations if the extension is not available.
"""

from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(
        "Geemeth/english_core.pyx",
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "nonecheck": False,
        },
        annotate=False,
    ),
)
