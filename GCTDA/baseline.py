import os
import numpy as np
from scipy.linalg import eigh
import igraph as ig
from sklearn.preprocessing import QuantileTransformer

def count_nodes(dataset_path):
    """
    For a given graph dataset, return a tuple (X,y), where:
    X contains the number of nodes of each graph
    y contains the labels of the graphs
    """
    y=[]
    X = []

    for f in sorted(os.listdir(dataset_path)):
        graph = ig.Graph.Read_Picklez(os.path.join(dataset_path,f))
        y.append(graph["label"])
        nb_vertices=len(graph.vs)
        X.append([nb_vertices])
    X = np.array(X).reshape(-1,1)
    return X,y

def histogram_degrees(dataset_path,bins=15):
    """
    For a given graph dataset, return a tuple (X,y), where:
    X contains the degree histogram of each graph
    y contains the labels of the graphs
    The degree histogram is the count of how many nodes have each degree.
    A quantile transformation is applied, to avoid most nodes being in the same
    bin if some nodes have a degree much larger than the others.
    """
    multisets = []
    y=[]
    for f in sorted(os.listdir(dataset_path)):
        graph = ig.Graph.Read_Picklez(os.path.join(dataset_path,f))
        y.append(graph["label"])
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
    return X,y

def laplacian_spectrum(dataset_path,max_size=200):
    """
    For a given graph dataset, return a tuple (X,y), where:
    X contains the eigenvalues of the normalized graph laplacian, in decreasing order.
    y contains the labels of the graphs

    The number of features returned is the minimum of max_size and of the number of nodes
    of the largest graph in the dataset. For smaller graphs, the feature vector is completed with 0s.
    """
    X = []
    y = []

    # Find the maximum size (number of vertices in a graph) in the dataset.
    # This will be the number of features 
    max_size_dataset = 0
    for f in sorted(os.listdir(dataset_path)):
        graph = ig.Graph.Read_Picklez(os.path.join(dataset_path,f))
        max_size_dataset = max(max_size_dataset, graph.vcount())
    max_size = min(max_size,max_size_dataset)

    # Compute eigenvalues of the graph laplacian
    for f in sorted(os.listdir(dataset_path)):
        graph = ig.Graph.Read_Picklez(os.path.join(dataset_path,f))
        y.append(graph["label"])
        L = graph.laplacian(normalized = True)
        egvals, _ = eigh(L)
        eigenvalues = np.zeros(max_size)
        eigenvalues[:min(max_size, graph.vcount())] = np.flipud(egvals)[:min(max_size, graph.vcount())]
        X.append(eigenvalues)
    return X,y

def heat_kernel_trace(dataset_path,T=np.logspace(-1, 1, 200)):
    """
    For a given graph dataset, return a tuple (X,y), where:
    X contains the heat kernel trace of the graphs.
    y contains the labels of the graphs
    """
    X = []
    y = []

    for f in sorted(os.listdir(dataset_path)):
        graph = ig.Graph.Read_Picklez(os.path.join(dataset_path,f))
        y.append(graph["label"])
        L = graph.laplacian(normalized=True)
        spectrum,_ = eigh(L)
        # Calculate the trace for each value in the array of times and sum
        # over the rows. This ensures that the resulting arrays are always
        # of the same cardinality.
        X.append(np.exp(-spectrum[:, np.newaxis] * T).sum(axis=0))
    return X,y