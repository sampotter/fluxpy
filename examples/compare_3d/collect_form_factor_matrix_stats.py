#!/usr/bin/env python

import json
import scipy.sparse
import sys

from flux.compressed_form_factors import CompressedFormFactorMatrix
from flux.model import compute_steady_state_temp
from flux.util import nbytes, tic, toc

from util import shape_model_from_obj_file

# read off paths from CLI arguments
planet = sys.argv[1]
FF_path = sys.argv[2] # SOURCES[0]
obj_path = sys.argv[3] # SOURCES[1]
T_path = sys.argv[4] # TARGETS[0]
T_time_path = sys.argv[5] # TARGETS[1]
p_path = sys.argv[6] # TARGETS[2]

# load compressed form factor matrix
if 'npz' in FF_path:
    FF = scipy.sparse.load_npz(FF_path)
else:
    FF = CompressedFormFactorMatrix.from_file(FF_path)

# load the shape model from the passed obj file
shape_model = shape_model_from_obj_file(obj_path)

# load planet parameters
with open('params.json', 'r') as f:
    params = json.load(f)[planet]

# get parameters needed for this experiment
sundir = np.array(params['sundir'])
F0 = params['F0']
alpha = params['albedo']
emiss = params['emissivity']

# compute insolation
Qdirect = shape_model.get_direct_irradiance(F0, sundir)

# compute steady state temperature
tic()
T = compute_steady_state_temp(FF, Qdirect, alpha, emiss)
T_time = toc()

# save temperature to disk
np.save(T_path, T)

# compute the number of visible entries per row in a "blocked"
# style... trying to avoid hitting the rails memory-wise here... we
# compute sqrt(N) MVPs at a time
block_size = int(np.ceil(np.sqrt(FF.num_faces)))
num_blocks = int(np.ceil(FF.num_faces/block_size))
num_vis = []
for block_index in range(num_blocks):
    i0, i1 = block_index*block_size, (block_index + 1)*block_size
    i1 = min(i1, FF.num_faces)
    E = np.zeros((FF.num_faces, i1 - i0), dtype=FF.dtype)
    for k, i in enumerate(range(i0, i1)):
        E[i, k] = 1
    assert(E.ravel().sum() == i1 - i0)
    FF_times_E = FF@E
    num_vis.extend((FF_times_E != 0).sum(axis=0).tolist())
num_vis = np.array(num_vis)

# compute visibility percent and save to disk
p = num_vis/FF.num_faces
np.save(p_path, p)

# load assembly time from disk
with open(T_time_path, 'w') as f:
    print(T_time, file=f)
