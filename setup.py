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
    version="1.0.0",
    author="Leonardo dos Santos",
    author_email="Leonardo.dosSantos@unige.ch",
    packages=["p_winds"],
    url="https://github.com/ladsantos/p-winds",
    license="MIT",
    description="Parker wind models for planetary atmospheres",
    install_requires=[line.strip() for line in
                      open('requirements.txt', 'r').readlines()],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ]
)
