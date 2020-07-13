import os
import numpy as np
from scipy.linalg import eigh
import igraph as ig
from sklearn.preprocessing import QuantileTransformer

def generate_baseline_features(graphs,method):
    if method == "count":
        X=count_nodes(graphs)
    elif method=="degrees":
        X = histogram_degrees(graphs)
    elif method == "spectrum":
        X = laplacian_spectrum(graphs)
    elif method=="heat_kernel_trace":
        X = heat_kernel_trace(graphs)
    else:
        raise ValueError("Unrecognized baseline method.")
    return X

def count_nodes(graphs):
    """
    For a given graph dataset, return the number of nodes of each graph
    """
    X = []
    for graph in graphs:
        nb_vertices=len(graph.vs)
        X.append([nb_vertices])
    X = np.array(X).reshape(-1,1)
    return X

def histogram_degrees(graphs,bins=15):
    """
    For a given graph dataset, return the degree histogram of each graph

    The degree histogram is the count of how many nodes have each degree.
    A quantile transformation is applied, to avoid most nodes being in the same
    bin if some nodes have a degree much larger than the others.
    """
    multisets = []
    for graph in graphs:
        multisets.append(graph.degree())
    values = np.concatenate(multisets)
    values = np.reshape(values,(-1,1))
    scaler = QuantileTransformer()
    scaler.fit(values)
    X=[]
    for multiset in multisets:
        scaled = scaler.transform(np.reshape(multiset,(-1,1)))
        hist = np.histogram(scaled,bins = bins, range=(0,1))[0]
        X.append(hist)
    return X

def laplacian_spectrum(graphs,max_size=200):
    """
    For a given graph dataset, return the eigenvalues of the normalized graph laplacian, in decreasing order.

    The number of features returned is the minimum of max_size and of the number of nodes
    of the largest graph in the dataset. For smaller graphs, the feature vector is completed with 0s.
    """
    X = []

    # Find the maximum size (number of vertices in a graph) in the dataset.
    # This will be the number of features 
    max_size_dataset = 0
    for graph in graphs:
        max_size_dataset = max(max_size_dataset, graph.vcount())
    max_size = min(max_size,max_size_dataset)

    # Compute eigenvalues of the graph laplacian
    for graph in graphs:
        L = graph.laplacian(normalized = True)
        egvals, _ = eigh(L)
        eigenvalues = np.zeros(max_size)
        eigenvalues[:min(max_size, graph.vcount())] = np.flipud(egvals)[:min(max_size, graph.vcount())]
        X.append(eigenvalues)
    return X

def heat_kernel_trace(graphs,T=np.logspace(-1, 1, 200)):
    """
    For a given graph dataset, return the heat kernel trace of the graphs.
    """
    X = []

    for graph in graphs:
        L = graph.laplacian(normalized=True)
        spectrum,_ = eigh(L)
        # Calculate the trace for each value in the array of times and sum
        # over the rows. This ensures that the resulting arrays are always
        # of the same cardinality.
        X.append(np.exp(-spectrum[:, np.newaxis] * T).sum(axis=0))
    return X