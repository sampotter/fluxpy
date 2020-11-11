'''This script is a port of the contents of erwan_spice.m (in the same
directory) to from MICE to spiceypy.

'''

import netCDF4
import numpy as np
import os
import spiceypy as spice

clktol = '10:000'

spice.kclear()
spice.furnsh('simple.furnsh')

# Define time window

utc0 = '2011 MAR 01 00:00:00.00'
utc1 = '2012 MAR 01 00:00:00.00'
stepet = 3600

et0 = spice.str2et(utc0)
et1 = spice.str2et(utc1)
nbet = int(np.ceil((et1 - et0)/stepet))
et = np.linspace(et0, et1, nbet)

# Sun positions over time period

possun = spice.spkpos('SUN', et, 'MOON_ME', 'LT+S', 'MOON')[0]
lonsun = np.arctan2(possun[:, 1], possun[:, 0])
lonsun = np.mod(180*lonsun/np.pi, 360)
radsun = np.sqrt(np.sum(possun[:, :2]**2, axis=1))
latsun = np.arctan2(possun[:, 2], radsun)
latsun *= 180/np.pi

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
    np.arcsin(np.cos(c)*np.sin(np.rad2deg(lat0)) +
              (np.cos(np.rad2deg(lat0))*np.sin(c)*grdy_)/rho))

lon_ = np.mod(
    lon0 + np.rad2deg(
        np.arctan2(
            grdx_*np.sin(c),
            np.cos(np.rad2deg(lat0))*rho*np.cos(c)
            - np.sin(np.rad2deg(lat0)*grdy_*np.sin(c)))),
    360)

lat_[(grdx_ == 0) & (grdy_ == 0)] = lat0
lon_[(grdx_ == 0) & (grdy_ == 0)] = lon0

# Go from lon/lat to cartesian

az_ = np.deg2rad(lon_)
el_ = np.deg2rad(lat_)
r_ = R + grdz_

xc_ = r_*np.cos(az_)*np.cos(el_)
yc_ = r_*np.sin(az_)*np.cos(el_)
zc_ = r_*np.sin(el_)
