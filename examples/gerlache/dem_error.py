import matplotlib.pyplot as plt
import numpy as np

plt.ion()

dem = np.lib.format.open_memmap(npy_path) # already in meters

h = 5e2 # cm (5mpp DEM)

Z = dem[10000:10100, 15000:15100] # km originally

dZ_dx_cm = (1e5*np.diff(Z, axis=0))/h
dZ_dy_cm = (1e5*np.diff(Z, axis=1))/h

plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.imshow(Z)
plt.title('DEM [km]')
plt.colorbar()
plt.subplot(1, 3, 2)
plt.imshow(dZ_dx_cm)
plt.title('$\partial z / \partial x$ [cm]')
plt.colorbar()
plt.subplot(1, 3, 3)
plt.imshow(dZ_dy_cm)
plt.title('$\partial z / \partial y$ [cm]')
plt.colorbar()
plt.show()

plt.figure(figsize=(8, 4))
plt.axvline(x=30, linewidth=1, zorder=1, c='k')
plt.axvline(x=55, linewidth=1, linestyle='--', zorder=1, c='k')
plt.axvline(x=-30, linewidth=1, zorder=1, c='k')
plt.axvline(x=-50, linewidth=1, linestyle='--', zorder=1, c='k')
plt.hist(dZ_dx_cm.ravel(), zorder=2, histtype='step',
         label=r'$\partial z / \partial x$ [cm]', bins=129)
plt.hist(dZ_dy_cm.ravel(), zorder=3, histtype='step',
         label=r'$\partial z / \partial y$ [cm]', bins=129)
plt.legend()
plt.gca()
plt.show()
