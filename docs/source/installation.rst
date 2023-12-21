Installation
============

``p-winds`` requires the packages ``numpy``, ``scipy``, ``astropy``, and
``flatstar`` for running its core calculations.

Option 1: Using ``pip`` (stable version)
--------------------------------------------------------------

Simply run the following command:

.. code-block:: bash

   pip install p-winds

Option 2: Compile from source (development version)
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

Download reference spectra and set environment variable
-----------------------------------------------------

If you want to use the function ``tools.generate_muscles_spectrum()`` or ``tools.standard_spectrum()``, you will need to download the reference data separately and set the environment variable ``$PWINDS_REFSPEC_DIR``. For your convenience, you can download all spectra supported by ``p-winds`` in `this compressed file
<https://stsci.box.com/s/0sz1grsc9jo0z7we4htos0fr4gcs13ks>`_.

After unzipping the compressed file, move the fits files to a path of your choosing; in this example, I will use the path ``/$HOME/Data/p-winds_reference_spectra``. Next, set an environment variable ``$PWINDS_REFSPEC_DIR`` that points to this path; this is done by running the following code in the command line:

.. code-block:: bash

   export $PWINDS_REFSPEC_DIR="/$HOME/Data/p-winds_reference_spectra"

If you do not want to set this environment variable every time you start a new session, you can add this line to your Record Columnar file (or ``rc``) in your user folder. Usually, this file is ``~/.bashrc`` if you use the bash shell, or ``~/.zshrc`` if you use the z-shell.