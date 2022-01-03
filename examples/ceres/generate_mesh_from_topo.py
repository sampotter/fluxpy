#!/usr/bin/env python

# processes grd file in native resolution (no interpolation)


import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import os
import scipy.interpolate
import pickle
#import meshio
#import trimesh

from flux.plot import tripcolor_vector
from flux.shape import get_centroids, get_surface_normals
from flux.shape import CgalTrimeshShapeModel as MyTrimeshShapeModel
#from flux.shape import EmbreeTrimeshShapeModel as MyTrimeshShapeModel


def mesh_from_xyz():
    pathfn = os.path.join('/arsia/Dawn/DWNCHSPG_2/DATA', 'tmp.xyz')
    
    a = np.loadtxt(pathfn)
    X = a[:,0]
    Y = a[:,1]
    Z = a[:,2]
    print('Number of vertices', a.shape[0] )
    print('x_min:',np.min(X), 'x_max:',np.max(X) )
    print('y_min:',np.min(Y), 'y_max:',np.max(Y) )
    print('z_min:',np.min(Z), 'z_max:',np.max(Z))
    print('loaded',pathfn)

    # generate (x,y)
    #points_mesh = np.array([X, Y]).T
    points_mesh = a[:,:2]
    
    # with native values of (x,y), Delaunay triangulates without interpolation
    triangulation = scipy.spatial.Delaunay(points_mesh)
    V, F = triangulation.points, triangulation.simplices

    V = np.row_stack([V.T, Z]).T
    assert np.linalg.norm(V-a) == 0.
    
    return V, F


def mesh_from_grd():
    # Load a DEM stored as a netCDF4 file, and pull out the coordinate data.
    pathfn = os.path.join('.', 'ingersoll41.grd')
    rootgrp = netCDF4.Dataset(pathfn)
    X = np.array(rootgrp.variables['x'])
    Y = np.array(rootgrp.variables['y'])
    Z = np.array(rootgrp.variables['z'])
    x0, x1 = rootgrp.variables['x'].actual_range
    y0, y1 = rootgrp.variables['y'].actual_range
    nx = rootgrp.dimensions['x'].size
    ny = rootgrp.dimensions['y'].size
    print('x_min:',x0, 'x_max:',x1, 'x_inc:',(x1-x0)/(nx-1), 'n_columns:',nx)
    print('y_min:',y0, 'y_max:',y1, 'y_inc:',(y1-y0)/(ny-1), 'n_rows:',ny)
    print('loaded',pathfn)

    # generate (x,y)
    X_mesh, Y_mesh = np.meshgrid(X,Y)
    points_mesh = np.array([X_mesh.flatten(), Y_mesh.flatten()]).T
    
    # with native values of (x,y), Delaunay triangulates without interpolation
    triangulation = scipy.spatial.Delaunay(points_mesh)
    V, F = triangulation.points, triangulation.simplices

    z = np.ndarray.flatten(Z)
    V = np.row_stack([V.T, z]).T
    
    return V, F
    

def mesh_from_icq():
    # Load a DEM stored as a icq file (Gaskell file format)
    pathfn = os.path.join('/home/norbert/Dawn/PSR2/', 'tmp.topo')
    a = np.loadtxt(pathfn)
    X = a[:,0]
    Y = a[:,1]
    Z = a[:,2]
    print('Number of vertices', a.shape[0] )
    print('x_min:',np.min(X), 'x_max:',np.max(X) )
    print('y_min:',np.min(Y), 'y_max:',np.max(Y) )
    print('z_min:',np.min(Z), 'z_max:',np.max(Z))
    print('loaded',pathfn)

    points_mesh = np.vstack((X,Y)).T
    
    # with native values of (x,y), Delaunay triangulates without interpolation
    delaunay = scipy.spatial.Delaunay(points_mesh, qhull_options = 'QJ')
    V, F = delaunay.points, delaunay.simplices
    assert V.shape[0] == a.shape[0]
    
    V = np.row_stack([V.T, Z]).T
    assert np.linalg.norm(V-a) == 0.
    
    return V, F


if __name__ == '__main__':

    #V, F = mesh_from_xyz()
    V, F = mesh_from_grd()
    #V, F = mesh_from_icq()

    num_faces = F.shape[0]
    num_vertices = V.shape[0]
    print('created mesh with %d triangles and %d vertices' % (num_faces, num_vertices) )
    
    # Make plot of topography
    fig, ax = tripcolor_vector(V, F, V[:,2], cmap='jet')
    fig.savefig('topo2.png')
    plt.close(fig)
    print('wrote topo2.png')

    # Let's use another Python library (meshio) to save the triangle
    # mesh as an OBJ file (optional).
    #points = V
    #cells = [('triangle', F)]
    #mesh = meshio.Mesh(points, cells)
    #mesh_path = 'mesh%d.obj' % num_faces
    #mesh.write(mesh_path)
    #print('wrote %s to disk' % mesh_path)
    
    P = get_centroids(V, F)
    N = get_surface_normals(V, F)
    N[(N*P)[:,2] < 0] *= -1  # normals must be outward
    #print( 'array shapes:', N.shape, P.shape, (N*P).shape )
    
    # Create a triangle mesh shape model using the vertices (V) and
    # face indices (F).
    shape_model = MyTrimeshShapeModel(V, F, N=N, P=P)

    # Write the mesh to disk as an OBJ file
    #trimesh.Trimesh(V, F).export('mesh.obj')
    #print('- wrote mesh.obj')

    # Write the mesh to disk as a numpy file
    #np.savez('mesh', shape_model=shape_model, V=V, F=F)
    #print('- wrote mesh.npy')
    
    # Write the mesh to disk as pickle file
    with open('mesh.bin', 'wb') as f:
        pickle.dump(shape_model,f)
        print('- wrote mesh.bin')

    # Optionally, output properties of the shape model
    print('x_min:',np.min(V[:,0]), 'x_max:',np.max(V[:,0]) )
    print('y_min:',np.min(V[:,1]), 'y_max:',np.max(V[:,1]) )
    print('z_min:',np.min(V[:,2]), 'z_max:',np.max(V[:,2]) )
    
