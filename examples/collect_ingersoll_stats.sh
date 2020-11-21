#!/usr/bin/env bash

for p in {5..10}
do
	echo "p = $p"
	OUTDIR="ingersoll_p$p"
	time ./ingersoll.py -p $p --outdir=$OUTDIR
	echo ""
done
