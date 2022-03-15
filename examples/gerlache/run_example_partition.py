import sys

import numpy as np
import submitit as submitit

from flux.compressed_form_factors import CompressedFormFactorMatrix, FormFactorPartitionBlock
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
    parts = None

    verts = np.load(f'gerlache_verts_{max_inner_area_str}_{max_outer_area_str}.npy')
    faces = np.load(f'gerlache_faces_{max_inner_area_str}_{max_outer_area_str}.npy')

    # convert verts from km to m
    verts *= 1e3

    normals = get_surface_normals(verts, faces)
    normals[normals[:, 2] > 0] *= -1

    shape_model = CgalTrimeshShapeModel(verts, faces, normals)

    # take 4 parts among faces
    parts = np.array_split(range(faces.shape[0]),4,axis=0)
    # parts = None
    print(parts)

    # use quadtree by default
    tic()
    # if parts is None:
    #     FF = CompressedFormFactorMatrix(
    #         shape_model, tol=tol, min_size=16384)
    # else:
    FF = CompressedFormFactorMatrix(
        shape_model, tol=tol, parts=parts, min_size=16384,
        RootBlock=FormFactorPartitionBlock, slurm=executor)
    assembly_time = toc()

    with open('FF_assembly_times.txt', 'a') as f:
        print(f'{tol_str} {max_inner_area_str} {max_outer_area_str} {assembly_time}', file=f)

    FF.save(f'FF_{max_inner_area_str}_{max_outer_area_str}_{tol_str}.bin')

exit()

    #
    # zip_files = glob.glob("/home/sberton2/Scaricati/Shackleton60mpp_pow2_*.zip")
    #
    # if True:
    #     for zip_file in zip_files:
    #         dirunzip = zip_file.split('/')[-1].split('.')[0]
    #         try:
    #             with zipfile.ZipFile(zip_file) as z:
    #                 z.extractall(f"in/{dirunzip}")
    #                 print(f"Extracted all to {dirunzip}")
    #         except:
    #             print("Invalid file")
    #
    #         meshes = glob.glob(f"in/{dirunzip}/Shackleton60mpp_pow2_*.stl")
    #         meshes = sorted(meshes)
    #
    #         # prepare mesh for FF (no ext)
    #         V, F = combine_meshes(meshes[1:], save_to=f"in/{dirunzip}_stereo.stl", verbose=False)
    #         Vcart = stereo_to_cart(V,-90,radius*1.e3)
    #         mesh_out = f"in/{dirunzip}.stl"
    #         mesh = meshio.Mesh(Vcart, [('triangle', F)])
    #         mesh.write(mesh_out)
    #
    #         # prepare total mesh for Sun occlusion (ext included)
    #         V, F = combine_meshes(meshes[1:] + meshes[:1], save_to=f"in/{dirunzip}_ext.stl", verbose=False)
    #         Vcart = stereo_to_cart(V,-90,radius*1.e3)
    #         mesh_out = f"in/{dirunzip}_ext.stl"
    #         mesh = meshio.Mesh(Vcart, [('triangle', F)])
    #         mesh.write(mesh_out)
    #
    # if False:
    #     executor.update_parameters(slurm_name="flux_FF")
    #     jobs = executor.map_array(make_FF,
    #                               [(f"in/{zip_file.split('/')[-1].split('.')[0]}.stl", 1e-1,
    #                                 f"in/{zip_file.split('/')[-1].split('.')[0]}_1e-1.bin") for zip_file in zip_files])
    #     for job in jobs:
    #         print(job.result())
    #
    # if True:
    #     executor.update_parameters(slurm_name="flux_spin")
    #     jobs = executor.map_array(spinup_model,
    #                               [(f"in/FF_{zip_file.split('/')[-1].split('.')[0]}_1e-1.bin", 20, 0)
    #                                for zip_file in zip_files])
    #     for job in jobs:
    #         print(job.result())
    #
    # for zip_file in zip_files:
    #     plot_Tmax(f"out/{zip_file.split('/')[-1].split('.')[0]}_1e-1/T_frames/",
    #               f"in/FF_{zip_file.split('/')[-1].split('.')[0]}_1e-1.bin", "1e-1")
    #
