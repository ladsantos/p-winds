#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

if sys.argv[-1] == "publish":
    os.system("python setup.py sdist upload")
    sys.exit()

setup(
    name="p_winds",
    version="2.0.1b",
    author="Leonardo dos Santos",
    author_email="ldsantos@stsci.edu",
    packages=["p_winds"],
    url="https://github.com/ladsantos/p-winds",
    license="MIT",
    description="Parker wind models for planetary atmospheres",
    install_requires=[line.strip() for line in
                      open('requirements.txt', 'r').readlines()],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ]
)
