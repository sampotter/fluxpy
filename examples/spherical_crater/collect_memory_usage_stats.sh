#!/usr/bin/env sh

PMAX=$1 # fixed value of p for now
CONTOUR_MODE=$2

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

mkdir -p ./mprof_stats

./collect_data.py --tol=1e-2 -p$PMAX --mprof=True --outdir=./mprof_stats/1e-2  $CONTOUR_ARGS
./collect_data.py            -p$PMAX --mprof=True --outdir=./mprof_stats/gt    $CONTOUR_ARGS
