#!/bin/sh
# Script used to evaluate the impact of the different filtrations, and of extended persistence
for dataset in MUTAG PROTEINS NCI1 DD BZR DHFR COX2 PTC-FM PTC-FR PTC-MR PTC-MM IMDB-BINARY IMDB-MULTI REDDIT-BINARY; do
	for filtration in degree; do
		for dim in 2; do
		for spread in 0.05 0.1 0.2 0.3 0.4 0.5; do
		for res in 3 5 7 9 11 13; do
		 bsub -n 4 -W 35 python evaluate.py -d $dataset -f $filtration --dim $dim --spread $spread --res $res -o results_res
		done
		done
		done
	done
done
