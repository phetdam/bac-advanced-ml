"""setup.py to build/install the bac-advanced-ml supporting code.

.. note::

   The "legacy" setup.py is still used to accomodate for C extensions.

.. codeauthor:: Derek Huang <djh459@stern.nyu.edu>
"""

from setuptools import find_packages, setup

from bacaml import __version__

# formal package name, different from the imported package name
_PACKAGE_NAME = "bac-advanced-ml"
# short description
_SHORT_DESC = (
    "A Python package containing lecture materials and exercises for the NYU "
    "Stern BAC Advanced Team's intro to machine learning curriculum."
)


def _setup():
    # get long description from README.rst
    with open("README.rst") as df:
        long_desc = df.read()
    # run setup
    setup(
        name=_PACKAGE_NAME,
        version=__version__,
        description=_SHORT_DESC,
        long_description=long_desc,
        long_description_content_type="text/x-rst",
        author="Derek Huang",
        author_email="djh458@stern.nyu.edu",
        license="MIT",
        url="https://github.com/phetdam/bac-advanced-ml",
        classifiers=[
            "License :: OSI Approved :: MIT License",
            "Operating System :: POSIX :: Linux",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8"
        ],
        project_urls={
            "Source": "https://github.com/phetdam/bac-advanced-ml"
        },
        python_requires=">=3.6",
        install_requires=[
            "numpy>=1.19.1",
            "scipy>=1.5.2",
            "scikit-learn>=0.23.2"
        ],
        extras_require={"tests": ["pytest>=6.0.1"]},
        packages=find_packages()
    )


if __name__ == "__main__":
    _setup()