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
pip install python-embree
```
Next, install the rest of the dependencies by running `pip install -r
requirements.txt`.

### Installing this package ###

Finally, run:
``` shell
pip install .
```
To install the package in editable mode, so that changes in this
directory propagate immediately, run:
``` shell
pip install -e .
```

## Running the unit tests ##

To run python-flux's unit tests, just run:
``` shell
./run_tests.sh
```
from the root of this repository. This script just runs `python -m unittest discover tests`; i.e., it discovers and runs all Python unit tests found in the directory `./tests`.

## Running the examples ##

The `examples` directory contains simple Python scripts illustrating
how to use this package. To run one of them, try:
``` shell
cd examples
python haworth.py
```
after following the installation instructions above. For more specific
instructions about particular examples, see below.

### Lunar south pole example

Running the following scripts in order:
``` shell
python lsp_make_shape_model.py
python lsp_form_factor_matrix.py
python lsp_spice.py
```
will do the following:

1. Create a shape model from a section of the DEM in the file
   `LDEM_80S_150M_adjusted.grd`. The resulting shape model will
   contain around ~160K faces. The script can be modified easily to
   change the portion of the DEM that is used to build the shape model
   or the number of triangles.

2. Build a compressed form factor matrix for the shape model
   constructed in the previous step and write it to disk.

3. Load both the shape model and the compressed form factor matrix and
   use SPICE (through SpiceyPy) to step through a number of sun
   positions, compute the direct insolation, and solve for the steady
   state temperature at each time instant. This temperature is plotted
   and written to disk.
