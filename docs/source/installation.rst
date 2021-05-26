Installation
============

``p-winds`` requires the packages ``numpy``, ``scipy``, ``astropy``, and
``flatstar`` for running its core calculations.

Option 1: Using ``pip`` (more up-to-date version, recommended)
--------------------------------------

Simply run the following command:

.. code-block:: bash

   pip install p-winds

Option 2: Using ``conda-forge`` (stable version)
------------------------------------------------

If you have Anaconda, the recommended way to install ``p-winds`` and its
dependencies is through the ``conda-forge`` channel.

.. code-block:: bash

   conda update conda
   conda install -c conda-forge p-winds

The ``conda-forge`` version might be a few numbers behind the `pip` and
development versions.

Option 3: Compile from source (development version)
---------------------------------------------------

First, clone the repository and navigate to it:

.. code-block:: bash

   git clone https://github.com/ladsantos/p-winds.git && cd p-winds

And then compile ``p-winds`` from source:

.. code-block:: bash

   python setup.py install

You can test the installation from source with ``pytest`` (you may need to
install ``pytest`` first):

.. code-block:: bash

   pytest tests
