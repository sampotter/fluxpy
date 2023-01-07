#!/usr/bin/env python3

import numpy as np
import spiceypy as spice

spice.kclear()
spice.furnsh('ceres.furnsh')
spice.furnsh('kernels/dawn_ceres_SPC151123.tpc')
#spice.furnsh('kernels/dawn_ceres_obliquity20deg.tpc') # overwrites previous tpc
#spice.furnsh('kernels/sb_ceres_grv_171219.bsp')

# Define time window

et0 = spice.str2et('2015 APR 23 00:00:00.00')
et1 = spice.str2et('2018 JUN 06 00:00:00.00')
et1 = et0 + 86400*1680.  # approx

# northern summer solstice, declination +4
#et0 = spice.str2et('2015 JUL 24 08:00:00.00')
#et1 = spice.str2et('2015 JUL 24 18:00:00.00')

# southern summer solstice, declination -4
#et0 = spice.str2et('2017 DEC 23 05:00:00.00')
#et1 = spice.str2et('2017 DEC 23 15:00:00.00')

# equinoxes, declination 0
#et0 = spice.str2et('2016 NOV 13 12:00:00.00')
#et1 = spice.str2et('2016 NOV 13 20:00:00.00')
#et0 = spice.str2et('2018 DEC 24 22:00:00.00')
#et1 = spice.str2et('2018 DEC 25 8:00:00.00')


et = np.linspace(et0, et1, 2000, endpoint=False)

# multipe days
#et = np.concatenate(
#    (np.linspace(spice.str2et('2015 JUL 24 08:00:00.00'), spice.str2et('2015 JUL 24 18:00:00.00'), 100, endpoint=False),
#     np.linspace(spice.str2et('2016 NOV 13 12:00:00.00'), spice.str2et('2016 NOV 13 20:00:00.00'), 100, endpoint=False),
#     np.linspace(spice.str2et('2017 DEC 23 05:00:00.00'), spice.str2et('2017 DEC 23 15:00:00.00'), 100, endpoint=False)
#     ))


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
    utc =  spice.et2utc(et[j], "C", 0) 
    print(et[j], np.rad2deg(latsun[j]), np.rad2deg(lonsun[j]), radsun[j], utc )
    #print(*sun_dirs[:,j])

