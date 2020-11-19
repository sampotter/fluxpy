#!/usr/bin/env bash

for p in {5..10}
do
	echo "p = $p"
	OUTDIR="ingersoll_p$p"
	./ingersoll.py -p $p --outdir=$OUTDIR
done
