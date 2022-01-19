#!/usr/bin/env python

import numpy as np
import sys

from osgeo import gdal

tif_path = sys.argv[1]
npy_path = tif_path[:-4] + '.npy'

ds = gdal.Open(tif_path)
band = ds.GetRasterBand(1)
arr = band.ReadAsArray()
np.save(npy_path, arr)
