import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np

plt.ion()

dem = np.lib.format.open_memmap(npy_path) # already in meters

Z = dem[10000:10100, 15000:15100] # m originally

dZ_y_cm = 1e2*np.diff(Z, axis=0)
dZ_x_cm = 1e2*np.diff(Z, axis=1)

vmax = max(abs(dZ_y_cm).ravel().max(), abs(dZ_x_cm).ravel().max())
vmin = -vmax

plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.imshow(Z, cmap=cc.cm.rainbow)
plt.title('DEM [km]')
plt.colorbar()
plt.subplot(1, 3, 2)
plt.imshow(dZ_x_cm, vmax=vmax, vmin=vmin, cmap=cc.cm.coolwarm)
plt.title('$\partial z$ in $x$ direction [cm]')
plt.colorbar()
plt.subplot(1, 3, 3)
plt.imshow(dZ_y_cm, vmax=vmax, vmin=vmin, cmap=cc.cm.coolwarm)
plt.title('$\partial z$ in $y$ direction [cm]')
plt.colorbar()
plt.show()

plt.figure(figsize=(8, 4))
plt.axvline(x=30, linewidth=1, zorder=1, c='k')
plt.axvline(x=55, linewidth=1, linestyle='--', zorder=1, c='k')
plt.axvline(x=-30, linewidth=1, zorder=1, c='k')
plt.axvline(x=-50, linewidth=1, linestyle='--', zorder=1, c='k')
plt.hist(dZ_x_cm.ravel(), zorder=2, histtype='step',
         label=r'$\partial z$ in $x$ direction [cm]', bins=129)
plt.hist(dZ_y_cm.ravel(), zorder=3, histtype='step',
         label=r'$\partial z$ in $y$ direction [cm]', bins=129)
plt.legend()
plt.gca()
plt.show()
