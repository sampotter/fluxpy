#!/usr/bin/env bash

# move these into a text file and read them from there
F0=1365
RHO=0.11
EMISS=0.95
EQUIL_T_SUN_AZ=0 # degrees [0, 360]
EQUIL_T_SUN_EL=95 # degrees [0, 180], 0 deg = at north pole, 180 = south

# test areas
MAX_INNER_AREAS=("3.0" "1.5" "0.75" "0.4")

# # uncomment for the real thing!
# MAX_INNER_AREAS=("3.0" "1.5" "0.75" "0.4" "0.2" "0.1" "0.05" "0.025")

MAX_OUTER_AREA="3.0" # very coarse

TOLS=(1e-1 1e-2)

# echo "making meshes"
# for MAX_INNER_AREA in "${MAX_INNER_AREAS[@]}"
# do
# 	echo "- max inner area: $MAX_INNER_AREA, max outer area: $MAX_OUTER_AREA"
# 	./make_mesh_sfp.py $MAX_INNER_AREA $MAX_OUTER_AREA
# done

echo "computing FF matrices"
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

# TODO: use raytracing to compute error plot (AND TABLES) for equilibrium temperature

# TODO: other plots AND TABLES:
# - FF time comparisons
# - FF size comparisons
# - FF direct error comparisons

# TODO: make time-dependent error plots AND TABLES

# TODO: make pointwise time-dependent error plot using rt
