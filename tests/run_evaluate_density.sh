#!/bin/sh
# Script used to evaluate baseline methods (distributions of node values)

for dataset in MUTAG PROTEINS NCI1 NCI109 DD BZR DHFR COX2 PTC-FM PTC-FR PTC-MR PTC-MM COLLAB IMDB-BINARY IMDB-MULTI REDDIT-BINARY REDDIT-MULTI-5K REDDIT-MULTI-12K watts-strogatz:0.1-0.2 BA-rewire:0.1-0.25; do
	for density in degree birth.degree*cc.degree degree*mean.degree degree+mean.degree degree+mean.degree+min.degree+max.degree+sum.degree edgeBetweenness cc.edgeBetweenness cycles.edgeBetweenness cc.edgeBetweenness+cycles.edgeBetweenness cc.edgeBetweenness+cycles.edgeBetweenness+degree+hks; do
		 	bsub -n 4 -W 3:55 python evaluate.py -d $dataset --density $density --spread 0.1 --res 10 -o results_density
	done
done
