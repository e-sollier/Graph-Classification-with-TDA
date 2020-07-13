# Graph Classification with TDA

This repository contains the code used for my Lab Rotation in the MLCB Lab at ETH Zürich. The goal was to use Persistent Homology for graph classification.

## Dependencies

A `pyproject.toml` file is provided, which means that you can install the dependencies using poetry. Note that it requires the pyper library, which for now is in a private repository.

## Quick Demo

The notebook [quick_demo.ipynb](https://github.com/e-sollier/Graph-Classification-with-TDA/blob/master/quick_demo.ipynb) can be used to easily generate persistence images and use them for classification with random forests. It can be used to evaluate the performance on one dataset for one set of parameters.

## Evaluating several methods on numerous datasets

The performance of each method is evaluated by running 10 times a 10-fold nested cross-validation on all the datasets. This takes a long time, especially for large datasets like REDDIT-MULTI-12K. That's why I performed the evaluations in parallel on a cluster (EULER). The script `evaluate.py` evaluates one method for one dataset and stores the result in a single file. I used simple shell scripts to submit this script with different sets of parameters. Then, the script `merge_results.py` can be used to aggregate all the results into a single csv file.

## Datasets

I downloaded graph datasets from https://chrsmrrs.github.io/datasets/docs/datasets/ and I converted them to pickle files in igraph format (one graph per file). The converted graphs are provided in tests/Datasets/preprocessed. It is also possible to use the script `convert_graphs.py` in tests/Datasets to convert any graph dataset downloaded from TUDatasets.
