# TEMPORARY CODE...

shape_model = FF.shape_model
V = shape_model.V
F = shape_model.F

grid = pv.UnstructuredGrid({vtk.VTK_TRIANGLE: F}, V)
grid['p'] = p

plotter = pvqt.BackgroundPlotter()
plotter.add_mesh(grid, scalars='p', cmap=cc.cm.bmw)

xmin, ymin, zmin = V.min(0)
xmax, ymax, zmax = V.max(0)
dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
xmid, ymid, zmid = xmin + dx/2, ymin + dy/2, zmin + dz/2

pos = np.array([xmid, ymid, zmin - dz])
look = np.array([0, 0, 1]).astype(np.float64)
up = np.array([0, 1, 0]).astype(np.float64)
m, n = 512, 512
shape = (m, n)
h = 1.05*min(dx, dy)/max(m, n)
p_grid = imray(shape_model, p, pos, look, up, shape, h=h)

extent = [xmid - h*shape[0]/2, xmid + h*shape[0]/2,
          ymid - h*shape[1]/2, ymid + h*shape[1]/2]

cmap = cc.cm.bmw
cmap.set_bad(color='black')

plt.figure()
plt.imshow(p_grid, interpolation='none', extent=extent, cmap=cmap)
plt.xlabel(r'$x$ [km]')
plt.ylabel(r'$y$ [km]')
plt.colorbar()
plt.show()
