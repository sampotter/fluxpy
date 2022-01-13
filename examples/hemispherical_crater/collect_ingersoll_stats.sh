#!/usr/bin/env bash

TOL=$1

mkdir -p stats/eps_$TOL

for p in {4..10}
do
	echo "p = $p"
	OUTDIR="stats/eps_$TOL/ingersoll_p$p"
	time ./ingersoll.py -p $p --tol=$TOL --outdir=$OUTDIR
	echo ""
done
