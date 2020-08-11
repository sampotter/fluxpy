# Fast radiosity Python tutorial

The tutorial is in `tutorial.py`, which depends on `form_factors.py`
and `lunar_south_pole_80mpp_curvature.grd`. To use the tutorial, you
first need to set up your Python environment, which can be done
easily.

## Set up a virtual environment and clone the repository

Make a new directory and clone this repository to it. Then, inside the
directory that you've just created, run `python -m venv .`. This will
create a "virtual environment", which is a useful tool that Python
provides for managing dependencies for projects. The new directory
"contains" the virtual environment.

## Activate the virtual environment

To activate the virtual environment, run `source bin/activate` from
the directory containing it. Make sure to do this before doing
anything else below.

## Getting the dependencies

First, install
[python-embree](https://github.com/sampotter/python-embree) in this
virtual environment:
``` shell
git clone https://github.com/sampotter/python-embree
pip install python-embree
```

Next, install the rest of the dependencies by running `pip install -r
requirements.txt`.

## Run the tutorial

Now you can just run:
``` shell
python tutorial.py
```
The script will omit some command-line output explaining what it's
doing and save some plots and files to disk. Examine the contents of
tutorial.py to follow along with what it's doing specifically.
