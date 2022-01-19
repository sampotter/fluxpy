#!/usr/bin/env bash

DEM_NAME="ldem_87s_5mpp"
DEM_GEOTIFF_PATH="$DEM_NAME.tif"
DEM_NPY_PATH="$DEM_NAME.npy"
DEM_URL="https://pgda.gsfc.nasa.gov/data/LOLA_5mpp/87S/ldem_87s_5mpp.tif"

if [ ! -f $DEM_NPY_PATH ]; then
	echo "Didn't find Gerlache DEM. Downloading from PGDA..."
	wget $DEM_URL

	echo "Converting from GeoTIFF to .npy..."
	./convert_geotiff_to_npy.py $DEM_GEOTIFF_PATH

	echo "Deleting GeoTIFF file."
	rm $DEM_GEOTIFF_PATH
fi
