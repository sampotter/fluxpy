# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

from setuptools import Extension, setup # , find_packages
from Cython.Build import cythonize

DISTNAME = 'python-flux'

ext_modules = [
    Extension(
        name='flux.thermal',
        sources=['flux/pcc/conductionQ.c', 'flux/pcc/tridag.c',
                 'flux/thermal.pyx'],
    )
]

setup(
    name=DISTNAME,
    packages=['flux'],
    ext_modules=cythonize(ext_modules, language_level=3)
)
