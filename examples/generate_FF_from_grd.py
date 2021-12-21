#!/usr/bin/env python

# generates topography-agnostic mesh and the form factor matrix
# processes grd file in native resolution (no interpolation)


cmap = dict()
try:
    import colorcet as cc
    cmap['jet'] = cc.cm.rainbow
    cmap['gray'] = cc.cm.gray
    cmap['fire'] = cc.cm.fire
except ImportError:
    print('failed to import colorcet: using matplotlib colormaps')
    cmap['jet'] = 'jet'
    cmap['gray'] = 'gray'
    cmap['fire'] = 'inferno'


import matplotlib.pyplot as plt
import meshio
import netCDF4
import numpy as np
import os
import scipy.interpolate
import time
import pickle
import trimesh

from flux.compressed_form_factors import CompressedFormFactorMatrix
from flux.plot import plot_blocks, tripcolor_vector
from flux.shape import CgalTrimeshShapeModel
from flux.form_factors import get_form_factor_matrix



if __name__ == '__main__':

    compress = True  # compressed or uncompressed FF

    # Load a DEM stored as a netCDF4 file, and pull out the coordinate data.
    path = os.path.join('.', 'ingersoll41.grd')
    rootgrp = netCDF4.Dataset(path)
    X = np.array(rootgrp.variables['x'])
    Y = np.array(rootgrp.variables['y'])
    Z = np.array(rootgrp.variables['z'])
    x0, x1 = rootgrp.variables['x'].actual_range
    y0, y1 = rootgrp.variables['y'].actual_range
    nx = rootgrp.dimensions['x'].size
    ny = rootgrp.dimensions['y'].size
    print('x_min:',x0, 'x_max:',x1, 'x_inc:',(x1-x0)/(nx-1), 'n_columns:',nx)
    print('y_min:',y0, 'y_max:',y1, 'y_inc:',(y1-y0)/(ny-1), 'n_rows:',ny)
    del rootgrp, x0, x1, y0, y1
    print('loaded',path)

    # Create function z = z(x, y) that linearly interpolates DEM data
    z = scipy.interpolate.interp2d(X, Y, Z)

    # generate (x,y) from grd file
    X_mesh, Y_mesh = np.meshgrid(X,Y)
    points_mesh = np.array([X_mesh.flatten(), Y_mesh.flatten()]).T
    # with native values of (x,y), Delaunay triangulates without interpolation
    delaunay = scipy.spatial.Delaunay(points_mesh)
    V, F = delaunay.points, delaunay.simplices

    V = np.row_stack([V.T, np.array([z(*v)[0] for v in V])]).T
    num_faces = F.shape[0]
    print('created mesh with %d triangles' % num_faces)

    # Let's use another Python library (meshio) to save the triangle
    # mesh as an OBJ file (optional).
    points = V
    cells = [('triangle', F)]
    mesh = meshio.Mesh(points, cells)
    mesh_path = 'mesh%d.obj' % num_faces
    mesh.write(mesh_path)
    print('wrote %s to disk' % mesh_path)

    # Make plot of topography
    fig, ax = tripcolor_vector(V, F, V[:,2], cmap=cmap['jet'])
    fig.savefig('topo2.png')
    plt.close(fig)
    print('wrote topo2.png')

    # Since Embree runs in single precision, there's no reason to use
    # double precision here.
    V = V.astype(np.float32)

    # Create a triangle mesh shape model using the vertices (V) and
    # face indices (F).
    shape_model = CgalTrimeshShapeModel(V, F)

    # Build the compressed form factor matrix. All of the code related
    # to this can be found in the "form_factors.py" file in this
    # directory.
    if compress:
        t0 = time.perf_counter()
        FF = CompressedFormFactorMatrix(
            shape_model, tol=np.finfo(np.float32).resolution)
        print('assembled form factor matrix in %f sec (%1.2f MB)' %
              (time.perf_counter() - t0, FF.nbytes/(1024**2),))
        del t0

        # Python makes it very easy to serialize object hierarchies and
        # write them to disk as binary files. We do that here to save the
        # compressed form factor matrix. We can reload it later if we
        # want, without having to first compute it (or load an OBJ file,
        # or set up Embree, or any of that stuff).
        FF.save('FF.bin')
        print('saved compressed form factor matrix to FF.bin')

        # Make plot of blocks in form factor matrix
        #fig, ax = plot_blocks(FF._root)
        #fig.savefig('blocks.png')
        #plt.close(fig)
        #print('wrote blocks.png')

    else:  # uncompressed
        # Write the mesh to disk as an OBJ file
        #trimesh.Trimesh(V, F).export('mesh.obj')
        #print('- wrote mesh.obj')

        #np.savez('mesh', shape_model=shape_model, V=V, F=F)
        #print('- wrote mesh.npy')

        with open('mesh.bin', 'wb') as f:
            pickle.dump(shape_model,f)

        print('calculating full form factor matrix')
        FF = get_form_factor_matrix(shape_model)
        with open('FF.bin', 'wb') as f:
            pickle.dump(FF,f)
            print('saved uncompressed form factor matrix to FF.bin')

    # exit()   # convenient test below

    # Define constants used in the simulation:
    e0 = np.deg2rad(10) # Solar elevation angle
    F0 = 1365 # Solar constant
    dir_sun = np.array([0, -np.cos(e0), np.sin(e0)]) # Direction of sun

    # Compute the direct irradiance and find the elements which are in shadow.
    E = shape_model.get_direct_irradiance(F0, dir_sun, unit_Svec=True)

    # Make plot of direct irradiance
    fig, ax = tripcolor_vector(V, F, E, cmap='gray')
    fig.savefig('E.png')
    plt.close(fig)
    print('wrote E.png')
