#!/usr/bin/env bash

# The compression tolerance
TOL=1e-2

# Collect statistics for the "groundtruth form factor matrix"
# (i.e. the original sparse form factor matrix without compression
# applied)
./collect_ingersoll_gt_stats.sh

# Collect statistics for our method (the compressed form factor
# matrix)
./collect_ingersoll_stats.sh $TOL

# Make plots from the collected statistics
mkdir paper_plots
./make_paper_plots
