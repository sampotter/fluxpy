import numpy as np
import sys

# Load the DEM as a memory-mapped file
dem_npy_path = sys.argv[1]
dem = np.lib.format.open_memmap(dem_npy_path)

# TODO: convert stereographic to Cartesian

# TODO: use meshpy to create a Delaunay mesh of a specific region

# TODO: save mesh to disk
