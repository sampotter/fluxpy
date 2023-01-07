#!/usr/bin/env python3

import numpy as np
from generalorbit import generalorbit, equatorial2sundir


NT = 100  # time steps

semia = 2.76750591
ecc = 0.07582  # orbital eccentricity
omega = np.deg2rad(301.)
eps = np.deg2rad(4.0)  # axis tilt
Trot = 360./952.055  # rotation period in Earth days

for j in range(0,NT):
    edays = j/NT*1680.
    [Ls, decl, radsun] = generalorbit(edays,semia,ecc,omega,eps)
    print(edays, np.rad2deg(decl), np.rad2deg(Ls), radsun )
    
    HA = np.fmod( edays/Trot, 1.) * 2 * np.pi
    sundir = equatorial2sundir(decl, HA)
    #print(*sundir)

    
# plot 'tmp1' u 2:($4/149.598e6) w lp,'tmp2' u 2:4 w lp
