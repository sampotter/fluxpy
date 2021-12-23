# Lunar south pole examples

## `haworth.py`

This example:

- loads a section of the Lunar south pole topography from
  `lunar_south_pole_80mpp_curvature.grd`,
- creates a triangle mesh from the GRD file using
  [dmsh](https://pypi.org/project/dmsh/),
- assembles a compressed form factor matrix using CGAL's raytracer,
- computes the insolation for one sun direction (i.e., treating
  the sun as a point source at infinite distance),
- computes the steady state temperature for the given insolation,
- makes several plots and writes them to disk.

Optionally, if `DO_3D_PLOTTING` is `True`, then it will use
[PyVista](https://www.pyvista.org/) to make an interactive 3D plot.

Note that if `RAYTRACING_BACKEND == 'embree'`, then it will use the
Embree raytracer. This is faster, but using Embree is still
experimental.
