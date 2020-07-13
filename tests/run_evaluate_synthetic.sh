#!/bin/sh
# Script used to evaluate the different filtrations on synthetic datasets
for alg in watts-strogatz BA-rewire; do
	for filtration in degree hks nodeBetweenness; do
		for dim in 4; do
		for rewire in 0.0-0.05 0.0-0.10 0.1-0.15 0.1-0.20 0.1-0.25; do
		 bsub -n 4 -W 45 python evaluate.py -d $alg:$rewire -f $filtration --dim $dim -o results_synthetic
		done
		done
	done
done
