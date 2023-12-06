p-winds: Documentation
======================

.. image:: https://img.shields.io/badge/GitHub-ladsantos%2Fp%E2%88%92winds-blue.svg?style=flat
    :target: https://github.com/ladsantos/p-winds
    :alt: GitHub
.. image:: https://readthedocs.org/projects/p-winds/badge/?version=latest
    :target: https://p-winds.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. image:: https://travis-ci.com/ladsantos/p-winds.svg?branch=main
    :target: https://travis-ci.com/ladsantos/p-winds
    :alt: Build
.. image:: https://coveralls.io/repos/github/ladsantos/p-winds/badge.svg?branch=main
    :target: https://coveralls.io/github/ladsantos/p-winds?branch=main
    :alt: Coverage
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4551621.svg
    :target: https://doi.org/10.5281/zenodo.4551621
    :alt: DOI

``p-winds`` is an open-source Python implementation of Parker wind models for
planetary atmospheres, more specifically their upper atmosphere or corona. A paper
describing ``p-winds`` was published in `Dos Santos et al. (2022)`_. The
code is largely based on the theoretical framework of `Oklopčić & Hirata
(2018)`_ and `Lampón et al. (2020)`_, which themselves based their work on the
stellar wind model of `Parker (1958)`_.

.. _Dos Santos et al. (2022): https://ui.adsabs.harvard.edu/abs/2022A%26A...659A..62D/abstract
.. _Oklopčić & Hirata (2018): https://ui.adsabs.harvard.edu/abs/2018ApJ...855L..11O/abstract
.. _Lampón et al. (2020): https://ui.adsabs.harvard.edu/abs/2020A%26A...636A..13L/abstract
.. _Parker (1958): https://ui.adsabs.harvard.edu/abs/1958ApJ...128..664P/abstract

In the current version, the code solves the stead-state ionization distribution
of hydrogen and helium around the planet based on its physical parameters and
the high-energy irradiation arriving at the planet. It also calculates the
wavelength-dependent transit depths in the metastable helium triplet at 1.083
microns, and some metal lines in the UV (such as O I and C II). More species will be implemented
in the future.

``p-winds`` contains and distributes data products from the `MUSCLES and Mega-MUSCLES treasury surveys`_. If you use the ``tools.generate_muscles_spectrum()`` function in your study, we highly encourage you to cite `France et al. (2016)`_, `Youngblood et al. (2016)`_, `Loyd et al. (2016)`_, `Wilson et al. (2021)`_ and `Behr et al. (2023)`_.

.. _MUSCLES and Mega-MUSCLES treasury surveys: https://archive.stsci.edu/prepds/muscles/
.. _France et al. (2016): http://adsabs.harvard.edu/abs/2016ApJ...820...89F
.. _Youngblood et al. (2016): http://adsabs.harvard.edu/abs/2016arXiv160401032Y
.. _Loyd et al. (2016): http://adsabs.harvard.edu/abs/2016arXiv160404776P
.. _Wilson et al. (2021): https://ui.adsabs.harvard.edu/abs/2021ApJ...911...18W/abstract
.. _Behr et al. (2023): https://ui.adsabs.harvard.edu/abs/2023AJ....166...35B/abstract

    Notice: As of version 1.4.5, ``p-winds`` does not include the MUSCLES data anymore to make the package leaner. You will need to download this data separately and set an environment variable containing the path to the data. Follow the :ref:`installation guide <installation>`.

If you want to use the code without installing it locally, you can always run it
on the cloud (see `this Google Colaboratory quickstart notebook
<https://colab.research.google.com/drive/1mTh6_YEgCRl6DAKqnmRp2XMOW8CTCvm7?usp=sharing>`_).
Otherwise, check out the :ref:`Installation <installation>` guide.

You can `contribute <https://github.com/ladsantos/p-winds>`_ to the project
using GitHub.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   api
   quickstart
   tidal_effects
   advanced_tutorial
   exospheric_metals
