#!/usr/bin/env bash

PMIN=5
PMAX=6

# CONTOUR_MODE=1 # contour rim and shadow
# CONTOUR_MODE=2 # contour rim
CONTOUR_MODE=3 # no contouring

# The compression tolerance
TOLS=(1e-1)

PAPER_PLOT_DIR=spherical_crater_plots

# Collect statistics for the "groundtruth form factor matrix"
# (i.e. the original sparse form factor matrix without compression
# applied)

./collect_ingersoll_gt_stats.sh $PMIN $PMAX $CONTOUR_MODE

# Collect statistics for our method (the compressed form factor
# matrix)
for TOL in "${TOLS[@]}"
do
	echo "tol = $TOL"
	./collect_ingersoll_stats.sh $PMIN $PMAX $TOL $CONTOUR_MODE
done

# Do comparisons between groundtruth results and results obtained
# using various compressed form factor matrices
for TOL in "${TOLS[@]}"
do
	echo "doing direct comparison for tol = $TOL"
	./do_direct_comparisons.py $TOL
done

# Make block plots
./make_block_plots.py

# Collect memory usage statistics
./collect_memory_usage_stats.sh $PMAX $CONTOUR_MODE

# Make plots from the collected statistics
mkdir $PAPER_PLOT_DIR
./make_paper_plots.py $PAPER_PLOT_DIR
./make_memory_usage_plots.py $PAPER_PLOT_DIR
./make_error_hists.py
