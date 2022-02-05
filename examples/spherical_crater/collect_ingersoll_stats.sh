#!/usr/bin/env bash

PMIN=$1
PMAX=$2
TOL=$3
CONTOUR_MODE=$4

echo $TOL

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

mkdir -p stats/eps_$TOL

for p in $(seq $PMIN $PMAX)
do
	echo "p = $p"
	OUTDIR="stats/eps_$TOL/ingersoll_p$p"
	time ./collect_data.py -p $p --tol=$TOL --outdir=$OUTDIR $CONTOUR_ARGS
	echo ""
done
