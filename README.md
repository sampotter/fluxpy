[![Build Status](https://app.travis-ci.com/sampotter/python-flux.svg?branch=master)](https://app.travis-ci.com/sampotter/python-flux)

# Fast radiosity in Python #

## Installation ##

If using Windows, first install Microsoft Visual Studio C++ Build Tools following 
the [directions at this link](https://github.com/bycloudai/InstallVSBuildToolsWindows). This includes
an installation of MSVC, Windows SDK, and C++ CMake tools for Windows followed by adding MSBuild Tools
to your system path.

### Setting up a virtual environment ###

Using [Anaconda](https://www.anaconda.com/),
a virtual environment management tool for Python, create and activate a new conda environment and download
pip via the anaconda channel by running:
``` shell
conda create --name radiosity
conda activate radiosity
conda install -c anaconda pip
```
Alternatively, run `python -m venv .` to create a virtual environment in standard Python without using Anaconda. Activating the virtual 
environment (via `conda activate radiosity` or `source bin/activate`) means that all python and pip operations will be
managed using the packages installed into the environment. Therefore, the command
must be run before proceeding to the next steps (and before running any code).

### Getting the dependencies ###

First, install
[python-embree](https://github.com/sampotter/python-embree) in this
conda environment. Begin by installing Embree from the [Embree website](https://www.embree.org/). Typically, the Embree binaries,
headers, and libraries will all be installed to C:\Program Files\Intel\Embree3 by default. Then, clone, build, and install python-embree
to the radiosity conda environment:
``` shell
git clone https://github.com/sampotter/python-embree
cd python-embree
python setup.py build_ext -I "/c/Program\ Files/Intel/Embree3/include" -L "/c/Program\ Files/Intel/Embree3/lib"
python setup.py install
cd ..
```

Next, clone this repository and install the rest of the dependencies, including Boost and CGAL.
``` shell
git clone https://github.com/sampotter/python-flux
cd python-flux
pip install -r requirements.txt
```
Install the latest versions of [Boost](https://www.boost.org/) 
and [CGAL](https://www.cgal.org/download.html) from the corresponding websites. Finally,
link the CGAL and boost directories in `python-flux/setup.py` (via the `include_dirs` argument of the `flux.cgal.aabb` extension).

### Installing this package ###

Finally, from the cloned repository, run:
``` shell
python setup.py install
```

## Running the unit tests ##

To run python-flux's unit tests, just run:
``` shell
./run_tests.sh
```
from the root of this repository. All this script does is execute `python -m unittest discover ./tests`; i.e., it discovers and runs all Python unit tests found in the directory `./tests`.

## Running the examples ##

The `examples` directory contains simple Python scripts illustrating
how to use this package. Each example directory contains a README with
more information about that particulra example.

For instance, try running:
``` shell
cd examples/lunar_south_pole
python haworth.py
```
after following the installation instructions
above.
