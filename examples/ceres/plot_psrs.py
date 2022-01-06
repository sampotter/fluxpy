#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


R = 470  # nominal radius of Ceres


def plotknownpoints():
    lats = [86.2, 77.6, 69.9, 79.0, 81.3]
    lons = [79.3,353.9,114.0,259.1,313.9]
    annot = ['NP4','NP7','NP5','NP26','NP19']
    for i in range(0,len(lats)):
        x = R * np.cos(np.deg2rad(lats[i])) * np.sin(np.deg2rad(lons[i]))
        y = - R * np.cos(np.deg2rad(lats[i])) * np.cos(np.deg2rad(lons[i]))
        plt.plot(x,y,'co',label=annot[i],markersize=2 )


def drawcoordsys():
    r2 = R/2. # = lat 60
    r1 = R*np.cos(np.deg2rad(85.))
    t = np.linspace(0, 2*np.pi, 100)

    for lati in range(85,60-1,-5):
        lat = np.deg2rad(lati)
        x = R*np.cos(lat)*np.cos(t)
        y = R*np.cos(lat)*np.sin(t)
        plt.plot(x,y,'k-',linewidth=1)

    for loni in range(0,360,30):
        lon = np.deg2rad(loni)
        plt.plot([r1*np.cos(lon), r2*np.cos(lon)],
                 [r1*np.sin(lon), r2*np.sin(lon)],'k-',linewidth=1)



a = np.load('emax.npz')

V = a['V']
F = a['F']
P = a['P']
areas = a['areas']
Emax = a['Emax']

kpsr = np.where(Emax==0)[0]
print('Total area:', areas.sum()  )
print('PSR area:', areas[kpsr].sum() )

vv = np.vstack( (V[:,0], V[:,1]) )  # same as V[:,:2].T
#vv = np.vstack( (V[:,1], -V[:,0]) ) # rotated
#print(V[:,:2].shape, vv.shape)


fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# Make plot of maximum direct irradiance
im = ax.tripcolor(*vv, F, Emax, cmap = 'gray', vmin=0, vmax=100)
fig.colorbar(im, ax=ax)
ax.set_aspect('equal')

# Plot PSRs in different color on top
kpsr = np.where(Emax==0)[0]
im2 = ax.tripcolor( *vv , F[kpsr], Emax[kpsr], cmap = 'jet_r')
ax.set_xlim( np.min(vv[0,:]), np.max(vv[0,:]) )
ax.set_ylim( np.min(vv[1,:]), np.max(vv[1,:]) )
fig.tight_layout()

# draw coordinate grid
drawcoordsys()
#plotknownpoints()

plt.show()

print('writing figure to disk')
fig.savefig('EmaxP.png')
plt.close(fig)

