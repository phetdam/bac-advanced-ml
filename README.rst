.. README.rst for bac_advanced_ml

.. image:: https://raw.githubusercontent.com/phetdam/bac_advanced_ml/master/
   bac_logo1_small.png
   :alt: BAC logo 1 small

A repository containing lectures and exercises for the BAC Advanced Team's
intro to machine learning curriculum.

The source code in the repository is structured within a Python package.


Contents
========

   Announcement:

   A lesson syllabus will be included soon.

The repo contains both ``.tex`` source for lectures and exercises in
``lessons`` as well as Python code containing reference implementations for
some of the coding-related exercises in ``bac_advanced_ml``. |ell1|

.. |ell1| image:: https://render.githubusercontent.com/render/math?math=\ell^1


Installing
==========

From source
-----------

After cloning the repository, ``cd`` into the repository root and run

.. code:: bash

   python3 setup.py build_ext --inplace && pip3 install .

Note that currently this repository doesn't have any C extension modules so the
``build_ext`` step should not do anything. As a note, building C extension
modules on your platform requires that you use the same compiler that your
Python interpreter was compiled with. On Linux the compiler is usually ``gcc``,
on Windows it is usually Microsoft Visual C++, and on Mac ``clang``. However,
the ``setup.py`` file will take care of all the details for you and you can
trust that it gets the rather complicated compiler invocations correct.

From PyPI
---------

Ideally, once the exercise materials are in a relatively stable state, rolling
releases to PyPI can be made to simplify the installation process. This is a
future concern, however, and a feature that is not yet available.


Compiling TeX source
====================

To compile the ``.tex`` files to PDF using the ``compile_tex.sh`` file, you
will need to have `TeX Live`__ installed on your system. Ubuntu users can use
``apt-get``, but more legwork is needed for Windows and Mac users.

If one is interested in compiling to PDF only a particular ``.tex`` file, call
``compile_tex.sh`` with the name of the ``.tex`` file as an argument or use a
graphical editor like `Texmaker`__ to view source and PDF side-by-side.

``compile_tex.sh`` uses the standard chain of compilation commands, i.e.
``pdflatex -> bibtex -> pdflatex -> pdflatex``. For detailed usage on how to
use ``compile_tex.sh``, run ``./compile_tex.sh --help`` for help output.

.. __: https://tug.org/texlive/

.. __: https://www.xm1math.net/texmaker/index.html