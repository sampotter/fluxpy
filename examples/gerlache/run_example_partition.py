import sys

import numpy as np
import submitit as submitit

from flux.compressed_form_factors import CompressedFormFactorMatrix, FormFactorPartitionBlock
from flux.quadtree import get_quadrant_order
from flux.shape import get_surface_normals, CgalTrimeshShapeModel
from flux.util import tic, toc

radius = 1737.4 # km

if __name__ == '__main__':

    # executor is the submission interface (logs are dumped in the folder)
    executor = submitit.AutoExecutor(folder="log_slurm")
    # set timeout in min, and partition for running the job
    executor.update_parameters(  # slurm_account='j1010',
        slurm_name="flux_part",
        slurm_cpus_per_task=10,
        slurm_nodes=1,
        slurm_time=60 * 99,  # minutes
        slurm_mem='90G',
        slurm_array_parallelism=5)

    max_inner_area_str = sys.argv[1]
    max_outer_area_str = sys.argv[2]

    tol_str = sys.argv[3]
    tol = float(tol_str)
    try:
        num_parts = int(sys.argv[4])
        print(num_parts)
    except:
        num_parts = None
        parts = None
        print("- No parts selected, going quadtree")

    verts = np.load(f'gerlache_verts_{max_inner_area_str}_{max_outer_area_str}.npy')
    faces = np.load(f'gerlache_faces_{max_inner_area_str}_{max_outer_area_str}.npy')

    # convert verts from km to m
    verts *= 1e3

    normals = get_surface_normals(verts, faces)
    normals[normals[:, 2] > 0] *= -1

    shape_model = CgalTrimeshShapeModel(verts, faces, normals)

    if num_parts != None:
        # split mesh in 4 equal sets of faces (row-wise)
        # parts = np.array_split(range(faces.shape[0]), num_parts, axis=0)
        # or follow quadtree scheme
        P = shape_model.P
        PI = P[:, :2] # get (x,y)
        parts = [I for I in get_quadrant_order(PI)]
        print(parts[0])
        # apply the same for each part in parts to get higher order division
        if num_parts == 4: # TODO extend for num_parts > 4
            parts2 = []
            for p in parts:
                JJ = [I for I in get_quadrant_order(PI[p])]
                parts2.extend([p[J] for J in JJ])
        elif num_parts > 2:
            print("Please use num_parts in [None, 2, 4]. Exit.")
            exit()

    # use quadtree by default
    tic()
    if parts is None:
        FF = CompressedFormFactorMatrix(
            shape_model, tol=tol, min_size=16384)
    else:
        FF = CompressedFormFactorMatrix(
            shape_model, tol=tol, parts=parts, min_size=16384,
            RootBlock=FormFactorPartitionBlock) #, slurm=executor)
    assembly_time = toc()

    with open('FF_assembly_times.txt', 'a') as f:
        print(f'{tol_str} {max_inner_area_str} {max_outer_area_str} {assembly_time}', file=f)

    FF.save(f'FF_{max_inner_area_str}_{max_outer_area_str}_{tol_str}.bin')
