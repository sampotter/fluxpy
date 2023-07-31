#!/usr/bin/env python
import matplotlib.pyplot as plt
import meshio
import meshpy.triangle as triangle
import numpy as np
import sys
import rioxarray as rio
import scipy.ndimage

path = 'ldem_60s_final_adj_240mpp_surf'
tif_path = path + '.tif'

max_inner_area_str = sys.argv[1]
max_outer_area_str = max_inner_area_str

R_roi_str = sys.argv[2]

max_inner_area = float(max_inner_area_str)
max_outer_area = float(max_outer_area_str)

Rp = 1737.4

rds = rio.open_rasterio(tif_path)

xgrid = rds.coords['y'].values*-1.e-3
ygrid = rds.coords['x'].values*1.e-3
dem = rds.data[0].T[:,::-1]
dem = scipy.ndimage.gaussian_filter(dem, sigma=5)

nx = len(xgrid)
ny = len(ygrid)

dx = rds.coords['x'].attrs["actual_range"][-1] - rds.coords['x'].attrs["actual_range"][0]
dy = rds.coords['y'].attrs["actual_range"][-1] - rds.coords['y'].attrs["actual_range"][0]
hx, hy = dx/nx, dy/ny

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
    # print("x,y",x,y)
    # start = time.time()
    dR = getz(x, y)
    # print("dR call took", time.time()-start)
    # print(dR)
    # start = time.time()
    # xi = xr.DataArray([x], dims="z")
    # yi = xr.DataArray([y], dims="z")
    # dR = np.squeeze(rds.interp(x=xi, y=yi).data)
    # print("rioxarray call took", time.time()-start)
    # print(dR)
    # exit()
    r = Rp + dR
    cos_el = np.cos(el)
    return r*np.array([cos_el*np.cos(az), cos_el*np.sin(az), np.sin(el)])

xc_roi, yc_roi = 8, -6
r_roi, R_roi = 10, int(R_roi_str)

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

verts_stereo = np.array(mesh.points)
verts = np.array([stereo2cart(*_) for _ in np.array(mesh.points)]).squeeze()
faces = np.array(mesh.elements)

print(f'  * {verts.shape[0]} vertices and {faces.shape[0]} faces')

area_str = f'{max_inner_area_str}_{R_roi_str}'
np.save(f'blurred_pole_verts_stereo_{area_str}', verts_stereo)
np.save(f'blurred_pole_verts_{area_str}', verts)
np.save(f'blurred_pole_faces_{area_str}', faces)
