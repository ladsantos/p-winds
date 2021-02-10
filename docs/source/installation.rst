Installation
============

``p-winds`` requires the packages ``numpy``, ``scipy``, and ``astropy`` for
running its main calculations. Installing ``astroquery`` and ``matplotlib`` is
also recommended.

The recommended way to install ``p-winds`` and its dependencies is through a
``conda`` environment and then compiling it from source. Installing it in an
environment is not strictly necessary (you can just skip to the compiling from
source if you prefer), but it is a good practice, especially because ``p-winds``
is currently an alpha release.

First, clone the repository:

.. code-block:: bash

   git clone https://github.com/ladsantos/p-winds.git

Now create the ``conda`` environment. This command will install all the
necessary dependencies and useful packages at their most up-to-date versions
inside the environment ``p_env``. Navigate to the source folder and execute:

.. code-block:: bash

   cd p-winds
   conda env create -f p-winds_environment.yml

Next, activate the ``p_env`` environment:

.. code-block:: bash

   conda activate p_env

Finally, compile ``p-winds`` from source:

.. code-block:: bash

   python setup.py install