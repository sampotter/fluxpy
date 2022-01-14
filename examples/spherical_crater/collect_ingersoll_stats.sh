#!/usr/bin/env bash

PMIN=$1
PMAX=$2
TOL=$3

echo $TOL

mkdir -p stats/eps_$TOL

for p in $(seq $PMIN $PMAX)
do
	echo "p = $p"
	OUTDIR="stats/eps_$TOL/ingersoll_p$p"
	time ./ingersoll.py -p $p --tol=$TOL --outdir=$OUTDIR
	echo ""
done
