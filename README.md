[![Build Status](https://app.travis-ci.com/sampotter/python-flux.svg?branch=master)](https://app.travis-ci.com/sampotter/python-flux)

# Fast radiosity in Python #

## Installation ##

### Set up a virtual environment and clone the repository ###

Make a new directory and clone this repository to it. Then, inside the
directory that you've just created, run `python -m venv .`. This will
create a "virtual environment", which is a useful tool that Python
provides for managing dependencies for projects. The new directory
"contains" the virtual environment.

### Activate the virtual environment ###

To activate the virtual environment, run `source bin/activate` from
the directory containing it. Make sure to do this before doing
anything else below.

### Getting the dependencies ###

First, install
[python-embree](https://github.com/sampotter/python-embree) in this
virtual environment:
``` shell
git clone https://github.com/sampotter/python-embree
cd python-embree
python setup.py install
cd -
```

Next, install the rest of the dependencies:
``` shell
git clone https://github.com/sampotter/python-flux
cd python-flux
pip install -r requirements.txt
```

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
