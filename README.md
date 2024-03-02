# p-winds

[![Documentation Status](https://readthedocs.org/projects/p-winds/badge/?version=latest)](https://p-winds.readthedocs.io/en/latest/?badge=latest) [![Build Status](https://travis-ci.com/ladsantos/p-winds.svg?branch=main)](https://travis-ci.com/github/ladsantos/p-winds)  [![arXiv](https://img.shields.io/badge/arXiv-2111.11370-b31b1b.svg)](https://arxiv.org/abs/2111.11370)
 [![Code DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4551621.svg)](https://doi.org/10.5281/zenodo.4551621)

> **Warning**: This is the development branch. Version 2.0 of `p-winds` implement self-consistent, fluid dynamics models instead of assuming isothermal Parker-wind models. If you would like to contribute, please submit a pull request to this branch.

Python implementation of Parker wind models for planetary atmospheres. `p-winds` produces simplified, 1-D models of the upper atmosphere of a planet, and perform radiative transfer to calculate observable spectral signatures. 

The scalable implementation of 1D isothermal models allows for atmospheric retrievals to calculate atmospheric escape rates and temperatures. In addition, the modular implementation allows for a smooth plugging-in of more complex descriptions to forward model their corresponding spectral signatures (e.g., self-consistent or 3D models).

As of version 2.0, `p-winds` also includes a Python wrapper for the self-consistent, hydrodynamic escape simulation code [ATES](https://github.com/AndreaCaldiroli/ATES-Code), originally developed by [Andrea Caldiroli](https://github.com/AndreaCaldiroli). See instructions on how to use it below.

Scientific background
---------------------
The isothermal models of `p-winds` are largely based on the theoretical framework of [Oklopčić & Hirata (2018)](https://ui.adsabs.harvard.edu/abs/2018ApJ...855L..11O/abstract) and [Lampón et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020A%26A...636A..13L/abstract), which themselves based their work on the stellar wind model of [Parker (1958)](https://ui.adsabs.harvard.edu/abs/1958ApJ...128..664P/abstract). A description about the implementation of tidal effects is discussed in [Vissapragada et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022AJ....164..234V/abstract).

A [paper describing `p-winds`](https://ui.adsabs.harvard.edu/abs/2022A%26A...659A..62D/abstract) (Dos Santos et al. 2022) and its usage for research-grade astronomical applications was published in the journal Astronomy & Astrophysics. If you use this code in your research, please consider citing it. If you use the ATES interface within the `fluid` module, please consider citing [Caldiroli et al. (2021)](https://ui.adsabs.harvard.edu/abs/2021A%26A...655A..30C/abstract).

`p-winds` contains and distributes data products from the [MUSCLES and Mega-MUSCLES treasury surveys](https://archive.stsci.edu/prepds/muscles/). If you use the `tools.generate_muscles_spectrum()` function in your study, we highly encourage you to cite [France et al. 2016](http://adsabs.harvard.edu/abs/2016ApJ...820...89F), [Youngblood et al. 2016](http://adsabs.harvard.edu/abs/2016arXiv160401032Y), [Loyd et al. 2016](http://adsabs.harvard.edu/abs/2016arXiv160404776P), [Wilson et al. 2021](https://ui.adsabs.harvard.edu/abs/2021ApJ...911...18W/abstract) and [Behr et al. 2023](https://ui.adsabs.harvard.edu/abs/2023AJ....166...35B/abstract).

Requirements
------------

`p-winds` requires the following packages:

* `python` versions 3.8 or later; the code has also been tested and validated in versions 3.6 (not supported) and 3.9.
* `numpy`
* `scipy` version 1.5 or later
* `astropy`
* [`flatstar`](https://github.com/ladsantos/flatstar)

If you wish to use the ATES wrapper in the `fluid` module, further requirements are necessary:
* A Fortran compiler: either `gfortran` or `ifort` needs to be available in your PATH
* The **custom version** of ATES forked from the [original](https://github.com/AndreaCaldiroli/ATES-Code) and available [here](https://github.com/ladsantos/ATES-Code/releases). The wrapper in the `fluid` module will not work with the original ATES code, only with the custom version.

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

### Download the custom ATES code and set environment variable

If you wish to use the ATES wrapper available in the `fluid` module, you will need to download a custom ATES code [here](https://github.com/ladsantos/ATES-Code/releases). The wrapper is not compatible with the original ATES code.

After downloading it, you will need to set the environment variable `$ATES_DIR` to the location of the ATES code in your computer. For this example, I will use `$HOME/ATES-Code`. This is done by running the following code in the command line:

```angular2html
export ATES_DIR="$HOME/ATES-Code"
```

If you do not want to set this environment variable every time you start a new session, you can add this line to your Record Columnar file (or `rc`) in your user folder. Usually, this file is `~/.bashrc` if you use a bash shell, or `~/.zshrc` if you use zshell.

### Download reference spectra and set environment variable

If you want to use the function `tools.generate_muscles_spectrum()` or `tools.standard_spectrum()`, you will need to download the reference data separately and set the environment variable `$PWINDS_REFSPEC_DIR`. For your convenience, you can download all spectra supported by `p-winds` in [this compressed file](https://stsci.box.com/s/0sz1grsc9jo0z7we4htos0fr4gcs13ks).

After unzipping the compressed file, move the fits files to a path of your choosing; in this example, I will use the path `$HOME/Data/p-winds_reference_spectra`. Next, set an environment variable `$PWINDS_REFSPEC_DIR` that points to this path; this is done by running the following code in the command line:

```angular2html
export PWINDS_REFSPEC_DIR="$HOME/Data/p-winds_reference_spectra"
```

If you do not want to set this environment variable every time you start a new session, you can add this line to your Record Columnar file (or `rc`) in your user folder. Usually, this file is `~/.bashrc` if you use a bash shell, or `~/.zshrc` if you use zshell. 

Quickstart example
------------------
Check out a quickstart [Google Colab Notebook here](https://colab.research.google.com/drive/1mTh6_YEgCRl6DAKqnmRp2XMOW8CTCvm7?usp=sharing). A similar quickstart Jupyter notebook is also available inside the `docs/source/` folder.

Contributing
------------
You can contribute to the development of ``p-winds`` either by submitting issues, or by submitting pull requests (PR). If you choose to submit a PR, please pull it to the ``dev`` branch, which is where the experiments happen before being merged to the ``main`` branch.

Future features and known problems
--------
Check out the [open issues](https://github.com/ladsantos/p-winds/issues).