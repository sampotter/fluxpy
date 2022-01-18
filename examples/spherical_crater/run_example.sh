#!/usr/bin/env bash

PMIN=5
PMAX=8

# The compression tolerance
TOLS=(1e-1 1e-2)

PAPER_PLOT_DIR=spherical_crater_plots

# Collect statistics for the "groundtruth form factor matrix"
# (i.e. the original sparse form factor matrix without compression
# applied)

./collect_ingersoll_gt_stats.sh $PMIN $PMAX

# Collect statistics for our method (the compressed form factor
# matrix)
for TOL in "${TOLS[@]}"
do
	echo "tol = $TOL"
	./collect_ingersoll_stats.sh $PMIN $PMAX $TOL
done

# Do comparisons between groundtruth results and results obtained
# using various compressed form factor matrices
for TOL in "${TOLS[@]}"
do
	echo "doing direct comparison for tol = $TOL"
	./do_direct_comparisons.py $TOL
done

# Collect memory usage statistics
./collect_memory_usage_stats.sh

# Make plots from the collected statistics
mkdir $PAPER_PLOT_DIR
./make_paper_plots.py $PAPER_PLOT_DIR
./make_memory_usage_plots.py $PAPER_PLOT_DIR
