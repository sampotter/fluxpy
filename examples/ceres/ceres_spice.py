#!/usr/bin/env python3

import numpy as np
import spiceypy as spice

spice.kclear()
spice.furnsh('simple.furnsh')
spice.furnsh('kernels/dawn_ceres_SPC151123.tpc')
spice.furnsh('kernels/sb_ceres_grv_171219.bsp')

# Define time window

et0 = spice.str2et('2015 APR 23 00:00:00.00')
et1 = spice.str2et('2018 JUN 06 00:00:00.00')

# northern summer solstice, declination +4
#et0 = spice.str2et('2015 JUL 24 08:00:00.00')
#et1 = spice.str2et('2015 JUL 24 18:00:00.00')

# southern summer solstice, declination -4
#et0 = spice.str2et('2017 DEC 23 05:00:00.00')
#et1 = spice.str2et('2017 DEC 23 15:00:00.00')

et = np.linspace(et0, et1, 100, endpoint=False)

# Sun positions over time period
possun = spice.spkpos('SUN', et, 'CERES_FIXED', 'LT+S', 'CERES')[0]

lonsun = np.arctan2(possun[:, 1], possun[:, 0])
lonsun = np.mod(lonsun, 2*np.pi)
radsun = np.sqrt(np.sum(possun[:, :3]**2, axis=1))
latsun = np.arctan2(possun[:, 2], radsun)

sun_dirs = np.array([
    np.cos(lonsun)*np.cos(latsun),
    np.sin(lonsun)*np.cos(latsun),
    np.sin(latsun)
]).T

sun_dirs = (possun[:,0], possun[:,1], possun[:,2]) / radsun

for j in range(0,len(et)):
    print(et[j], np.rad2deg(latsun[j]), np.rad2deg(lonsun[j]), radsun[j] )


