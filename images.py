import os
import igraph as ig
from pyper.persistent_homology import calculate_persistence_diagrams
from extended_persistence import compute_extended_persistence_diagrams
from filtrations import calculate_filtration, scale_filtration
import persim
import matplotlib.pyplot as plt
import numpy as np


def generate_img_dataset(dataset,filtration="degree",extended=False,dimensions=[0],spread=1,pixels=[7,7]):
    """From a graph dataset, generate a list of persistence images (flattened),
    and the associated graph labels

    Parameters
    ----------
    dataset:
        Name of the folder corresponding to a dataset. Ex: MUTAG
    filtration:
        Method to compute the filtration. Must be degree, jaccard or ricci.
    extended:
        If True, use extended persistence. Otherwise, ordinary persistence is used.
    dimensions:
        When extended is False, specifies the dimensions of the components to use.
        Can be [0] for connected components, [1] for cycles or [0,1] for both.
    spread:
        Standard deviation of gaussian kernel
    pixels:
        Resolution of the persistence image
    Returns
    -------
    images:
        list of flattened persistent images
    y:
        list of labels (used for classification)
    """

    path = "Datasets/preprocessed/"+dataset+"/"

    # Load the graphs and compute the filtration
    graphs=[]
    y=[]
    for f in sorted(os.listdir(path)):
        graph = ig.Graph.Read_Picklez(os.path.join(path,f))
        graph = calculate_filtration(graph,method=filtration,attribute_out="f")
        graphs.append(graph)
        y.append(graph["label"])

    # Scale the filtration values, so that they are between 0 and 1
    graphs = scale_filtration(graphs,"f",individual=False)
    
    # Compute persistence diagrams
    # There are 4 diagrams in case of extended persistence. Otherwise, it depends on dimensions.
    persistence_diagrams = [[] , [] ,[] , []] if extended else [[] for dim in dimensions]
    for graph in graphs:
        if extended:
            diagrams = compute_extended_persistence_diagrams(graph,attribute="f")
            for i in range(4):
                persistence_diagrams[i].append(diagrams[i])
        else:
            pd_0, pd_1 = calculate_persistence_diagrams(graph,vertex_attribute='f', edge_attribute='f')
            for i, dim in enumerate(dimensions):
                if dim==0:
                    persistence_diagrams[i].append(pd_0._pairs)
                else:
                    persistence_diagrams[i].append(pd_1._pairs)

    # Compute the persistence images
    # It is better to compute all the persistence images at once, because that way the
    # same scaling is applied to all the persistence diagrams.
    images = []
    for dgs in persistence_diagrams:
        pim = persim.PersImage(spread=spread, pixels=pixels, verbose=False,weighting_type="uniform")
        images.append(pim.transform(dgs))
    flattened_images = [np.concatenate([images[ind][i].flatten() for ind in range(len(persistence_diagrams))]) for i in range(len(images[0]))]
    
    return flattened_images,y
