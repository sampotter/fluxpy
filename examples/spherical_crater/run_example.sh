#!/usr/bin/env bash

PMIN=5
PMAX=8

# The compression tolerance
TOL=1e-2

# Collect statistics for the "groundtruth form factor matrix"
# (i.e. the original sparse form factor matrix without compression
# applied)

./collect_ingersoll_gt_stats.sh $PMIN $PMAX

# Collect statistics for our method (the compressed form factor
# matrix)
./collect_ingersoll_stats.sh $PMIN $PMAX $TOL

# Do comparisons between groundtruth results and results obtained
# using various compressed form factor matrices
./do_direct_comparisons.py

# Collect memory usage statistics
./collect_memory_usage_stats.sh

# Make plots from the collected statistics
mkdir paper_plots
./make_paper_plots.py
./make_memory_usage_plots.py
