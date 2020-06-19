#!/bin/sh
for dataset in MUTAG PROTEINS NCI1 NCI109 COLLAB IMDB-BINARY IMDB-MULTI REDDIT-BINARY REDDIT-MULTI-5K REDDIT-MULTI-12K; do
	for filtration in degree node_betweenness; do
		for dim in 1 2 4; do
			bsub -n 4 python evaluate.py -d $dataset -f $filtration --dim $dim -o results
		done
	done
done
