#!/usr/bin/env python

import netCDF4
import numpy as np
import os
import scipy.interpolate
import scipy.ndimage
import scipy.spatial

from flux.shape import get_centroids, get_surface_normals

# Read GRD file

path = os.path.join('.', 'LDEM_80S_150M_adjusted.grd')
rootgrp = netCDF4.Dataset(path)

grdx = np.array(rootgrp['x'])
grdy = np.array(rootgrp['y'])

grdz_ = np.array(rootgrp['z']).astype(np.float64)
grdz_ = grdz_.T # TODO: why is this necessary?

del rootgrp

grdx_, grdy_ = np.meshgrid(grdx, grdy)

# Unprotect from stereographic to lon/lat

lon0, lat0, R = 0, -90, 1737.4

rho = np.sqrt(grdx_**2 + grdy_**2)
c = 2*np.arctan2(rho, 2*R)

lat_ = np.rad2deg(
    np.arcsin(np.cos(c)*np.sin(np.deg2rad(lat0)) +
              (np.cos(np.deg2rad(lat0))*np.sin(c)*grdy_)/rho))

lon_ = np.mod(
    lon0 + np.rad2deg(
        np.arctan2(
            grdx_*np.sin(c),
            np.cos(np.deg2rad(lat0))*rho*np.cos(c)
            - np.sin(np.deg2rad(lat0))*grdy_*np.sin(c))),
    360)

lat_[(grdx_ == 0) & (grdy_ == 0)] = lat0
lon_[(grdx_ == 0) & (grdy_ == 0)] = lon0

# Go from lon/lat to cartesian

az_ = np.deg2rad(lon_)
el_ = np.deg2rad(lat_)
r_ = R + grdz_

i0, i1 = 600, 850
j0, j1 = 650, 975

az = az_[i0:i1, j0:j1]
el = el_[i0:i1, j0:j1]
r = r_[i0:i1, j0:j1]

x = r*np.cos(az)*np.cos(el)
y = r*np.sin(az)*np.cos(el)
z = r*np.sin(el)

# Make a simple mesh

I = np.arange(i0, i1)
J = np.arange(j0, j1)
IJ = np.array([_.flatten() for _ in np.meshgrid(I, J, indexing='ij')]).T

V = np.array([x.ravel(), y.ravel(), z.ravel()]).T
F = scipy.spatial.Delaunay(IJ).simplices

# Compute face normals by computing

P = get_centroids(V, F)
N = get_surface_normals(V, F)
N[(N*P).sum(1) < 0] *= -1

# Save the mesh data to disk as numpy binary files

np.save('haworth_V', V)
print('- wrote haworth_V.bin')

np.save('haworth_F', F)
print('- wrote haworth_F.bin')

np.save('haworth_N', N)
print('- wrote haworth_N.bin')
