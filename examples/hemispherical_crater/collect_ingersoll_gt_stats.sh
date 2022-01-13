#!/usr/bin/env bash

mkdir -p stats/gt

for p in {4..10}
do
	echo "p = $p"
	OUTDIR="stats/gt/ingersoll_p$p"
	time ./ingersoll.py -p $p --outdir=$OUTDIR
	echo ""
done
