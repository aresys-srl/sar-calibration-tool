Installation
============


Requirements
------------
The packages required by SAR Calibration Tool are specified in the file ``environment.yml`` included in its distribution.

Requirements:

* python 3.8


Install
-------
In order to install SAR Calibration Tool follow these steps:

Clone the SAR Calibration Tool ``git`` repository in a folder of your local hard drive and move inside it

Create a conda environment with python 3.8, then activate the environment

.. code-block:: bash
    $ conda create -n sct_env python=3.8
    $ activate sct_env

Install the package via pip install 

Install the coda package (see https://github.com/conda-forge/coda-feedstock)

Compile the external tool solid by going to the sct/external of your SCT installation and running

.. code-block:: bash

    $ f2py -c build_solid/solid__for_sct.for -m solid

N.B. You may have to first run (see https://github.com/numpy/numpy/issues/22572)

.. code-block:: bash

    $ export CFLAGS=-std=c99


Additional installation steps
-----------------------------
In order to execute the Jupyter Notebook included in SAR Calibration Tool distribution, some additional packages shall be installed in the SCT environment.

In a conda command window, activate the SCT environment and type the following instructions:

.. code-block:: bash

    $ pip install notebook
    $ pip install matplotlib
    $ pip install seaborn

In order to generate the tool documentation, install also:

.. code-block:: bash

    $ pip install -U sphinx
    $ pip install nbsphinx

Then the documentation can be generated moving to the ``docs`` folder and typing the following instruction:

.. code-block:: bash

    $ make.bat html

Support tools: solid
^^^^^^^^^^^^^^^^^^^^
SAR Calibration Tool makes use of an external tool called `solid <https://geodesyworld.github.io/SOFTS/solid.htm>`_.

This tool, originally developed in Fortran language, has been compiled to be imported in Python as a standard package and two libraries have been included in the distribution, one for Windows and one for Linux OS.

Such libraries are supposed to work in the SCT environment.

In case of issues it is suggested to recompile directly the Fortran source code (``solid__for_sct.for`` file included as well in the distribution) on the local machine. In order to do this, in a conda command window, activate the SCT environment and type the following instructions:

.. code-block:: bash

    $ conda install -c conda-forge fortran-compiler
    $ f2py -c <path to solid__for_sct.for file> -m solid

From Python code, the just created solid library can then be imported adding:

.. code-block:: bash

    from solid import solid
