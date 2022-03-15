#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


R = 470  # nominal radius of Ceres

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
fig.colorbar(im, ax=ax, label='Maximum incident flux (W/m^2)')
ax.set_aspect('equal')

# Plot PSRs in different color on top
kpsr = np.where(Emax==0)[0]
im2 = ax.tripcolor( *vv , F[kpsr], Emax[kpsr], cmap = 'jet_r')
ax.set_xlim( np.min(vv[0,:]), np.max(vv[0,:]) )
ax.set_ylim( np.min(vv[1,:]), np.max(vv[1,:]) )
fig.tight_layout()

# draw lines of equal latitude
levs = np.deg2rad(np.arange(60,90,5))
lat = np.arctan2( V[:,2] , np.sqrt(V[:,0]**2 + V[:,1]**2) )
ax.tricontour( *vv, F, lat, levs, colors = 'k')


plt.show()

print('writing figure to disk')
fig.savefig('EmaxP.png')
plt.close(fig)

