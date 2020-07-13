#!/bin/sh
# Script used to evaluate the impact of the different filtrations, and of extended persistence
for dataset in MUTAG PROTEINS NCI1 NCI109 DD BZR DHFR COX2 PTC-FM PTC-FR PTC-MR PTC-MM COLLAB IMDB-BINARY IMDB-MULTI REDDIT-BINARY REDDIT-MULTI-5K REDDIT-MULTI-12K; do
	for filtration in degree jaccard ricci hks nodeBetweenness; do
		for dim in 2 4; do
		 	bsub -n 4 -W 3:45 python evaluate.py -d $dataset -f $filtration --dim $dim -o results_filtration
		done
	done
done
