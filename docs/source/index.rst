p-winds: Documentation
======================

.. image:: https://img.shields.io/badge/GitHub-ladsantos%2Fp%E2%88%92winds-blue.svg?style=flat
    :target: https://github.com/ladsantos/p-winds

``p-winds`` is an open-source Python implementation of Parker wind models for
planetary atmospheres, more specifically their upper atmosphere or corona. The
code is largely based on the theoretical framework of `Oklopčić & Hirata
(2018)`_ and `Lampón et al. (2020)`_, which themselves based their work on the
stellar wind model of `Parker (1958)`_.

.. _Oklopčić & Hirata (2018): https://ui.adsabs.harvard.edu/abs/2018ApJ...855L..11O/abstract
.. _Lampón et al. (2020): https://ui.adsabs.harvard.edu/abs/2020A%26A...636A..13L/abstract
.. _Parker (1958): https://ui.adsabs.harvard.edu/abs/1958ApJ...128..664P/abstract

In the current version, the code solves the stead-state ionization distribution
of hydrogen around the planet based on its physical parameters and the
high-energy irradiation arriving at the planet.

If you want to use the code without installing it locally, you can always run it
on the cloud (see `this Google Colaboratory quickstart notebook
<https://colab.research.google.com/drive/1mTh6_YEgCRl6DAKqnmRp2XMOW8CTCvm7?usp=sharing>`_).
Otherwise, check out the :ref:`installation` guide, or the :ref:`quickstart`
Jupyter Notebook.

You can `contribute <https://github.com/ladsantos/p-winds>`_ to the project
using GitHub.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   api
   quickstart
