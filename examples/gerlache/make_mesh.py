import numpy as np
import sys
import pyvista as pv

from raster2mesh.mesh_manip import pcd2mesh
from raster2mesh.raster_roi import roi2pcd

raster_path = sys.argv[1]
mesh_resolution = int(sys.argv[2])
roi_coords = [int(x) for x in list(sys.argv[3].split(','))]

mesh_path = raster_path[:-4] + '.obj'

# open raster and select region to mesh-ify
# generate mesh of selected region
pcd = roi2pcd(raster_in=raster_path, raster_out="roi.tif", roi=roi_coords)
pcd2mesh(pcd, mesh_out=mesh_path, resolution=mesh_resolution)

# read and visualize resulting mesh
grid = pv.read(mesh_path)
grid.plot(show_scalar_bar=True, show_axes=True)

