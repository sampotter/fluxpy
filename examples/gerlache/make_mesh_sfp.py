#!/usr/bin/env python

import colorcet as cc
import meshpy.triangle as triangle
import numpy as np
import pyvista as pv
import pyvistaqt as pvqt
import sys
import vtk

from osgeo import gdal
from parula import parula_cmap

path = 'ldem_87s_5mpp'
npy_path = path + '.npy'
tif_path = path + '.tif'

max_inner_area_str = sys.argv[1]
max_outer_area_str = sys.argv[2]

max_inner_area = float(max_inner_area_str)
max_outer_area = float(max_outer_area_str)

Rp = 1737.4 # average radius of the moon at the south pole in meters,
            # I guess?

dataset = gdal.Open(tif_path)
dem = np.lib.format.open_memmap(npy_path) # already in meters

x0, x1 = map(int, dataset.GetMetadata()['x#actual_range'][1:-1].split(','))
y0, y1 = map(int, dataset.GetMetadata()['y#actual_range'][1:-1].split(','))

dx, dy = x1 - x0, y1 - y0
nx, ny = dem.shape # NOTE: not sure whether x is axis 0 or 1
hx, hy = dx/nx, dy/ny

# points are at pixel centers (see comment on PGDA:
# https://pgda.gsfc.nasa.gov/products/81)
xgrid = np.linspace(x0 + hx/2, x1 - hx/2, nx)
ygrid = np.linspace(y0 + hy/2, y1 - hy/2, ny)

xmin, xmax = xgrid.min(), xgrid.max()
ymin, ymax = ygrid.min(), ygrid.max()

scale_x2ind = (nx - 1)/(xmax - xmin)
scale_y2ind = (ny - 1)/(ymax - ymin)

def getdem(i, j):
    return dem[i, j]/1e3 # we want this in km initially

def getz(x, y):
    '''linearly interpolate z from memory-mapped DEM'''

    assert xmin <= x <= xmax and ymin <= y <= ymax

    isub = scale_x2ind*(x - xmin)
    jsub = scale_y2ind*(y - ymin)

    i0, i1 = int(np.floor(isub)), int(np.ceil(isub))
    j0, j1 = int(np.floor(jsub)), int(np.ceil(jsub))

    # handle special cases first

    if i0 == i1 and j0 == j1:
        return getdem(i0, j0)

    p1 = (y - ygrid[j0])/hy
    assert 0 <= p1 <= 1
    if i0 == i1:
        return (1 - p1)*getdem(i0, j0) + p1*getdem(i0, j1)

    p0 = (x - xgrid[i0])/hx
    assert 0 <= p0 <= 1
    if j0 == j1:
        return (1 - p0)*getdem(i0, j0) + p0*getdem(i1, j0)

    assert i1 == i0 + 1 and j1 == j0 + 1

    z00 = getdem(i0, j0)
    z01 = getdem(i0, j1)
    z10 = getdem(i1, j0)
    z11 = getdem(i1, j1)
    z0 = (1 - p0)*z00 + p0*z10
    z1 = (1 - p0)*z01 + p0*z11
    z = (1 - p1)*z0 + p1*z1

    return z # km

def unproject_stereographic(x, y):
    az0, el0, R = 0, -np.pi/2, Rp
    if x == 0 and y == 0:
        return az0, el0
    rho = np.sqrt(x**2 + y**2)
    c = 2*np.arctan2(rho, 2*R)
    c_c, s_c = np.cos(c), np.sin(c)
    c_l, s_l = np.cos(el0), np.sin(el0)
    az = az0 + np.arctan2(x*s_c, c_l*rho*c_c - s_l*y*s_c)
    az = np.mod(az, 2*np.pi)
    el = np.arcsin(c_c*s_l + c_l*y*s_c/rho)
    return az, el

def stereo2cart(x, y):
    az, el = unproject_stereographic(x, y)
    dR = getz(x, y)
    r = Rp + dR
    cos_el = np.cos(el)
    return r*np.array([cos_el*np.cos(az), cos_el*np.sin(az), np.sin(el)])

xc_roi, yc_roi = 0, -45
r_roi, R_roi = 20, 40

x0_roi, x1_roi = xc_roi - R_roi, xc_roi + R_roi
y0_roi, y1_roi = yc_roi - R_roi, yc_roi + R_roi

def should_refine(verts, _):
    verts = np.array(verts)
    assert verts.shape[0] == 3

    P = np.empty((3, 3))
    for i, (x, y) in enumerate(verts):
        P[i] = stereo2cart(x, y)

    tri_area = np.linalg.norm(np.cross(P[1] - P[0], P[2] - P[0]))/2

    xm, ym = np.mean(verts, 0)
    rm = np.sqrt((xm - xc_roi)**2 + (ym - yc_roi)**2)

    if rm < r_roi:
        max_area = max_inner_area
    else:
        max_area = ((rm - r_roi)/(R_roi - r_roi))*max_inner_area \
            + (rm/(R_roi - r_roi))*max_outer_area

    return tri_area > max_area

points = np.array(
    [(x0_roi, y0_roi), (x1_roi, y0_roi), (x1_roi, y1_roi), (x0_roi, y1_roi)],
    dtype=np.float32)
facets = np.array([(0, 1), (1, 2), (2, 3), (3, 0)], dtype=np.intc)

info = triangle.MeshInfo()
info.set_points(points)
info.set_facets(facets)

mesh = triangle.build(info, refinement_func=should_refine)

verts = np.array([stereo2cart(*_) for _ in np.array(mesh.points)]).squeeze()
faces = np.array(mesh.elements)

# # uncomment the following to plot:

# centroids = verts[faces].mean(1)
# centroids_stereographic = np.array(mesh.points)[faces].mean(1)

# grid = pv.UnstructuredGrid({vtk.VTK_TRIANGLE: faces}, verts)
# grid['dR'] = np.array([getz(x, y) for x, y in zip(*centroids_stereographic.T)])

# plotter = pvqt.BackgroundPlotter()
# plotter.add_mesh(grid, scalars='dR', cmap=parula_cmap)
# plotter.add_mesh(grid, show_edges=True)

np.save(f'gerlache_verts_{max_inner_area_str}_{max_outer_area_str}', verts)
np.save(f'gerlache_faces_{max_inner_area_str}_{max_outer_area_str}', faces)
