#!/usr/bin/env bash

mkdir -p stats/gt

PMIN=$1
PMAX=$2

for p in $(seq $PMIN $PMAX)
do
	echo "p = $p"
	OUTDIR="stats/gt/ingersoll_p$p"
	time ./ingersoll.py -p $p --outdir=$OUTDIR
	echo ""
done
