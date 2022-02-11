#!/usr/bin/env bash

MAX_INNER_AREAS=("1.5" "0.75" "0.4")
MAX_OUTER_AREA="3.0" # very coarse
TOLS=(1e-1 1e-2)

if [ ! -f ldem_87s_5mpp.tif ]; then
	echo "didn't find DEM. downloading it from PGDA..."
	wget https://pgda.gsfc.nasa.gov/data/LOLA_5mpp/87S/ldem_87s_5mpp.tif

fi

if [ ! -f ldem_87s_5mpp.npy ]; then
	echo "converting DEM to npy format..."
	./convert_geotiff_to_npy.py ldem_87s_5mpp.tif
fi

mkdir -p gerlache_plots

echo "making meshes"
for MAX_INNER_AREA in "${MAX_INNER_AREAS[@]}"
do
	echo "- max inner area: $MAX_INNER_AREA, max outer area: $MAX_OUTER_AREA"
	./make_mesh.py $MAX_INNER_AREA $MAX_OUTER_AREA
done

echo "computing FF matrices"
rm -f FF_assembly_times.txt
for MAX_INNER_AREA in "${MAX_INNER_AREAS[@]}"
do
	echo "- max inner area: $MAX_INNER_AREA, max outer area: $MAX_OUTER_AREA"
	./make_true_form_factor_matrix.py $MAX_INNER_AREA $MAX_OUTER_AREA
	echo "  * computed true FF matrix"
	for TOL in "${TOLS[@]}"
	do
		./make_compressed_form_factor_matrix.py $MAX_INNER_AREA $MAX_OUTER_AREA $TOL
		echo "  * computed compressed FF matrix (tol = $TOL)"
	done
done

# collect data to make equilibrium temperature plots
./collect_equil_T_data.py

# make equilibrium temperature plots
./make_equil_T_plots.py

# make plots comparing form factor matrices
./make_FF_comparison_plots.py

# collect data for time-dependent plots
echo "collecting time-dependent data"
for MAX_INNER_AREA in "${MAX_INNER_AREAS[@]}"
do
	echo "- max inner area: $MAX_INNER_AREA, max outer area: $MAX_OUTER_AREA"
	./collect_time_dep_data.py $MAX_INNER_AREA $MAX_OUTER_AREA true

	for TOL in "${TOLS[@]}"
	do
		./collect_time_dep_data.py $MAX_INNER_AREA $MAX_OUTER_AREA $TOL
	done
done

# make time-dependent plots
./make_time_dep_plots.py

# make block matrix plots
./make_block_plots.py
