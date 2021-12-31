#!/usr/bin/env python

import numpy
import matplotlib.pyplot as plt

a = numpy.load('emax.npz')

V = a['V']
F = a['F']
P = a['P']
areas = a['areas']
Emax = a['Emax']

kpsr = numpy.where(Emax==0)[0]
print('')
print('Total area:', areas.sum()  )
print('PSR area:', areas[kpsr].sum() )


fig, ax = plt.subplots(1, 1, figsize=(12, 10))

#vv = numpy.vstack( (V[:,0], V[:,1]) )  # same as V[:,:2]
vv = numpy.vstack( (V[:,1], -V[:,0]) ) # rotated
#print(V[:,:2].shape, vv.shape)

# Make plot of maximum direct irradiance
im = ax.tripcolor(*vv, F, Emax, cmap = 'gray', vmin=0, vmax=100)
fig.colorbar(im, ax=ax)
ax.set_aspect('equal')

# Plot PSRs in different color on top
kpsr = numpy.where(Emax==0)[0]
im2 = ax.tripcolor( *vv , F[kpsr], Emax[kpsr], cmap = 'jet_r')
ax.set_xlim( numpy.min(vv[0,:]), numpy.max(vv[0,:]) )
ax.set_ylim( numpy.min(vv[1,:]), numpy.max(vv[1,:]) )
fig.tight_layout()

plt.show()
fig.savefig('EmaxP.png')
plt.close(fig)

