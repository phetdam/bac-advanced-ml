# setup.py to build/install the bac_advanced_ml supporting code. note that the
# "legacy" setup.py is still used; i might want to build a C extension

from setuptools import find_packages, setup

# package name
_PACKAGE_NAME = "bac_advanced_ml"


def _setup():
    # get version
    with open("VERSION") as vf:
        version = vf.read().strip()
    # short and long descriptions
    short_desc = (
        "A Python package containing lecture materials and exercises for the "
        "Stern BAC Advanced Team's intro to machine learning curriculum."
    )
    with open("README.rst") as df:
        long_desc = df.read()
    # run setup
    setup(
        name = _PACKAGE_NAME,
        version = version,
        description = short_desc,
        long_description = long_desc,
        long_description_content_type = "text/x-rst",
        author = "Derek Huang",
        author_email = "djh458@stern.nyu.edu",
        license = "MIT",
        url = "https://github.com/phetdam/bac_advanced_ml",
        classifiers = [
            "License :: OSI Approved :: MIT License",
            "Operating System :: POSIX :: Linux",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8"
        ],
        project_urls = {
            "Source": "https://github.com/phetdam/bac_advanced_ml"
        },
        python_requires = ">=3.6",
        install_requires = [
            "numpy>=1.19", "scikit-learn>=0.23.2", "scipy>=1.5.2"
        ],
        packages = find_packages()
    )


if __name__ == "__main__":
    _setup()