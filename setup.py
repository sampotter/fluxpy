# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

from setuptools import Extension, setup # , find_packages
from Cython.Build import cythonize

DISTNAME = 'python-flux'

ext_modules = [
    Extension(
        name='flux.thermal',
        sources=['src/flux/thermal.pyx',
                 'src/flux/pcc/conductionQ.c',
                 'src/flux/pcc/conductionT.c',
                 'src/flux/pcc/tridag.c'],
        include_dirs=['.'],
        language='c'
    ),
    Extension(
        name='flux.view_factor',
        sources=['src/flux/view_factor/view_factor.pyx',
                 'src/flux/view_factor/view_factor_narayanaswamy.c'],
        include_dirs=['.'],
        language='c'
    ),
    Extension(
        name='flux.cgal.aabb',
        sources=['src/flux/cgal/aabb.pyx',
                 'src/flux/cgal/aabb_wrapper.cpp'],
        include_dirs=['.'],
        language='c++',
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp']
    )
]

setup(
    name=DISTNAME,
    packages=['flux'],
    package_dir={'': 'src'},
    ext_modules=cythonize(
        ext_modules,
        language_level=3,
        compiler_directives={'embedsignature': True}
    )
)
