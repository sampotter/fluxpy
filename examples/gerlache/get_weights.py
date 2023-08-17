import numpy as np
import scipy.sparse
import scipy

from flux.shape import CgalTrimeshShapeModel, get_surface_normals

from flux.util import nbytes, get_sunvec

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--max_inner_area', type=float, default=0.8)
parser.add_argument('--max_outer_area', type=float, default=3.0)

parser.set_defaults(feature=False)

args = parser.parse_args()


max_inner_area_str = str(args.max_inner_area)
max_outer_area_str = str(args.max_outer_area)

verts = np.load(f'gerlache_verts_{max_inner_area_str}_{max_outer_area_str}.npy')
faces = np.load(f'gerlache_faces_{max_inner_area_str}_{max_outer_area_str}.npy')

# convert verts from km to m
verts *= 1e3

normals = get_surface_normals(verts, faces)
normals[normals[:, 2] > 0] *= -1

shape_model = CgalTrimeshShapeModel(verts, faces, normals)


# Construct a weight matrix based on direct irradiance in the shape model
utc0 = '2011 MAR 01 00:00:00.00'
utc1 = '2011 MAR 30 00:00:00.00'
num_frames = 29*4
stepet = (3600*24*29)/num_frames
sun_vecs = get_sunvec(utc0=utc0, utc1=utc1, stepet=stepet, path_to_furnsh="simple.furnsh",
                      target='SUN', observer='MOON', frame='MOON_ME')

D = sun_vecs/np.linalg.norm(sun_vecs, axis=1)[:, np.newaxis]
D = D.copy(order='C')

E_t = []
for t in range(D.shape[0]):
    E_t.append(shape_model.get_direct_irradiance(1365, D[t]))
E_t = np.array(E_t)
weights = E_t.mean(axis=0)

np.save(f'weights_{max_inner_area_str}_{max_outer_area_str}.npy', weights)
scipy.io.savemat(f'weights_{max_inner_area_str}_{max_outer_area_str}.mat', {'weights':weights})