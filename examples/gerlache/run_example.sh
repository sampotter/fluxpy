#!/usr/bin/env bash

DEM_NAME="ldem_87s_50mpp"
DEM_GEOTIFF_PATH="$DEM_NAME.tif"
DEM_OBJ_PATH="$DEM_NAME.obj"
#DEM_URL="https://pgda.gsfc.nasa.gov/data/LOLA_5mpp/87S/ldem_87s_5mpp.tif"
DEM_URL="./ldem_87s_50mpp.tif"

if [ ! -f $DEM_OBJ_PATH ]; then
  # I loaded here the 50mpp version for the moment
#  echo "Didn't find Gerlache DEM. Downloading from PGDA..."
#  wget -O $DEM_GEOTIFF_PATH $DEM_URL

  echo "Converting from GeoTIFF to .obj..." # one shouldn't use .obj format, it's deprecated
  python make_mesh.py $DEM_GEOTIFF_PATH 250 -66536,-21239,-22038,32281 # I pre-selected a reasonable roi. Remove last arg to manually select on plot

  echo "Deleting GeoTIFF file."
#  rm $DEM_GEOTIFF_PATH
fi

echo "Convert mesh to cartesian..."
python unproject_mesh.py $DEM_OBJ_PATH

# compute FF

# get Sun position from SPICE and illuminate