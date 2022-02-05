#!/usr/bin/env bash

mkdir -p stats/gt

PMIN=$1
PMAX=$2
CONTOUR_MODE=$3

if [ $CONTOUR_MODE -eq 1 ]; then
	CONTOUR_ARGS="--contour_rim=True --contour_shadow=True"
elif [ $CONTOUR_MODE -eq 2 ]; then
	CONTOUR_ARGS="--contour_rim=True"
elif [ $CONTOUR_MODE -eq 3 ]; then
	CONTOUR_ARGS=""
else
	echo "contour mode should be 1, 2, or 3 (got: $CONTOUR_MODE)" 1>&2
	exit 1
fi

for p in $(seq $PMIN $PMAX)
do
	echo "p = $p"
	OUTDIR="stats/gt/ingersoll_p$p"
	time ./collect_data.py -p $p --outdir=$OUTDIR $CONTOUR_ARGS
	echo ""
done
