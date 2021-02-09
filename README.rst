.. README.rst for bac_advanced_ml

.. image:: https://raw.githubusercontent.com/phetdam/bac_advanced_ml/master/
   bac_logo1_small.png
   :alt: BAC logo 1 small

A repository containing lectures and exercises for the BAC Advanced Team's
intro to machine learning curriculum.

The source code in the repository is structured within a Python package.


Contents
========

TBA. This repository is a work in progress.


Installing
==========

From source
-----------

Installing the Python package from source is relatively straightforward. After
cloning the repository, ``cd`` into the repository root, and then simply run
in your shell

.. code:: bash

   python3 setup.py build_ext --inplace && pip3 install .

Note that currently this repository doesn't have any C extension modules so the
``build_ext`` step should not do anything. As a note, building C extension
modules on your platform requires that you use the same compiler that your
Python interpreter was compiled with. On Linux the compiler is usually ``gcc``,
on Windows it is usually Microsoft Visual C++, and on Mac ``clang``. However,
the ``setup.py`` file will take care of all the details for you and you can
trust that it gets the sometimes magical compiler invocations correct.

From PyPI
---------

Ideally, once the exercise materials are in a relatively stable state, rolling
releases to PyPI can be made to simplify the installation process. This is a
future concern, however, and won't be available for this semester.


Compiling TeX source
====================

To compile the ``.tex`` files using the ``compile_tex.sh`` file, you will need
to have `TeX Live`__ installed on your system. For Ubuntu users this can be
downloaded using ``apt-get`` while it can take a bit more legwork for Windows
and Mac users.

.. __: https://tug.org/texlive/