#!/bin/sh
# Script used to evaluate the impact of the linear weight compared to the uniform one,
# and what happens if we add the distribution of the degrees.
# For this script, it is required to remove the "random noise", so that points with 0
# persistence really have 0 persistence.
for dataset in MUTAG PROTEINS NCI1 NCI109 DD BZR DHFR COX2 PTC-FM PTC-FR PTC-MR PTC-MM COLLAB IMDB-BINARY IMDB-MULTI REDDIT-BINARY REDDIT-MULTI-5K REDDIT-MULTI-12K; do
	for filtration in degree; do
		for dim in 1 2; do
		for density in degree none; do
		for weight in uniform linear; do
		 bsub -n 4 -W 3:30 python evaluate.py -d $dataset -f $filtration --dim $dim -w $weight --density $density -o results_weights
		done
		done
		done
	done
done
