# p-winds

[![Documentation Status](https://readthedocs.org/projects/p-winds/badge/?version=latest)](https://p-winds.readthedocs.io/en/latest/?badge=latest) [![Build Status](https://travis-ci.com/ladsantos/p-winds.svg?branch=main)](https://travis-ci.com/ladsantos/p-winds) [![Coverage Status](https://coveralls.io/repos/github/ladsantos/p-winds/badge.svg?branch=main)](https://coveralls.io/github/ladsantos/p-winds?branch=main) [![arXiv](https://img.shields.io/badge/arXiv-2111.11370-b31b1b.svg)](https://arxiv.org/abs/2111.11370)
 [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4551621.svg)](https://doi.org/10.5281/zenodo.4551621)

Python implementation of Parker wind models for planetary atmospheres. **So many p's!** (Hence the name for the code.)

The main objective of this code is to produce simplified, 1-D models of the upper atmosphere of a planet, and perform radiative transfer to calculate observable spectral signatures. 

The scalable implementation of 1D models allows for atmospheric retrievals to calculate atmospheric escape rates and temperatures. In addition, the modular implementation allows for a smooth plugging-in of more complex descriptions to forward model their corresponding spectral signatures (e.g., self-consistent or 3D models).

A [paper describing `p-winds`](https://arxiv.org/abs/2111.11370) and its usage for research-grade astronomical applications has been accepted for publication in the journal Astronomy & Astrophysics. If you use this code in your research, please consider citing it.

Background
----------
`p-winds` is largely based on the theoretical framework of [Oklopčić & Hirata (2018)](https://ui.adsabs.harvard.edu/abs/2018ApJ...855L..11O/abstract) and [Lampón et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020A%26A...636A..13L/abstract), which themselves based their work on the stellar wind model of [Parker (1958)](https://ui.adsabs.harvard.edu/abs/1958ApJ...128..664P/abstract).

Installation
------------

You can install `p-winds` using `pip` or by compiling it from source.

### Option 1: Using `pip` (stable version)

Simply run the following command:
```angular2html
pip install p-winds
```

### Option 2: Compile from source (development version)

First, clone the repository and then navigate to it:
```angular2html
git clone https://github.com/ladsantos/p-winds.git
cd p-winds
```

And then compile it from source:
```angular2html
python setup.py install
```

You can test the installation from source with ``pytest`` (you may need to
install ``pytest`` first):
```angular2html
pytest tests
```

Quickstart example
------------------
Check out a quickstart [Google Colab Notebook here](https://colab.research.google.com/drive/1mTh6_YEgCRl6DAKqnmRp2XMOW8CTCvm7?usp=sharing). A similar quickstart Jupyter notebook is also available inside the `docs/source/` folder.

Contributing
------------
You can contribute to the development of ``p-winds`` either by submitting issues, or by submitting pull requests (PR). If you choose to submit a PR, please pull it to the ``dev`` branch, which is where the experiments happen before being merged to the ``main`` branch.

Future features and known problems
--------
Check out the [open issues](https://github.com/ladsantos/p-winds/issues).