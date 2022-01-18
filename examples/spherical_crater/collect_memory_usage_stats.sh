#!/usr/bin/env sh

p=10 # fixed value of p for now

mkdir -p ./mprof_stats

./ingersoll.py --tol=1e-2 -p$p --mprof=True --outdir=./mprof_stats/1e-2
./ingersoll.py            -p$p --mprof=True --outdir=./mprof_stats/gt
