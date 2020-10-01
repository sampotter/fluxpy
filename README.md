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

## Running the examples ##

The `examples` directory contains simple Python scripts illustrating
how to use this package. To run one of them, try:
``` shell
cd examples
python haworth.py
```
after following the installation instructions above.
