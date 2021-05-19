Installation
============


Requirements
------------
The packages required by SAR Calibration Tool are specified in the file ``environment.yml`` included in its distribution.

Minimum requirements:

* python 3.5 or higher
* numpy 1.17 or higher


Install
-------
In order to install SAR Calibration Tool follow these steps:

1. Clone the SAR Calibration Tool ``git`` repository in a folder of your local hard drive and move inside it

2. In a conda command window, type the following instruction, which creates the proper environment:
	
.. code-block:: bash

    $ conda env create --file environment.yml

3. In the same conda command window, type the following instruction, which activates the created environment:

.. code-block:: bash

    $ conda activate sct_env

4. | Install the SAR Calibration Tool package using ``pip`` tool:
   | (note: use the ``-e`` option to install it in ``edit`` mode)

.. code-block:: bash

    $ pip install .
    <or>
    $ pip install -e .


Additional installation steps
-----------------------------
In order to execute the Jupyter Notebook included in SAR Calibration Tool distribution, some additional packages shall be installed in the ``sct_env`` environment.

In a conda command window, activate the ``sct_env`` environment and type the following instructions:

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

Such libraries are supposed to work in the ``sct_env`` environment.

In case of issues it is suggested to recompile directly the Fortran source code (``solid__for_sct.for`` file included as well in the distribution) on the local machine. In order to do this, in a conda command window, activate the ``sct_env`` environment and type the following instructions:

.. code-block:: bash

    $ conda install -c conda-forge fortran-compiler
    $ f2py -c <path to solid__for_sct.for file> -m solid

From Python code, the just created solid library can then be imported adding:

.. code-block:: bash

    from solid import solid
