import os
import numpy as np
import matplotlib.pyplot as plt
import igraph as ig
import persim
from pyper.persistent_homology import calculate_persistence_diagrams

from GCTDA.extendedPersistence import compute_extended_persistence_diagrams
from GCTDA.filtrations import calculate_filtration, scale_filtration_quantile


def generate_img_dataset(dataset_path,filtration="degree",extended=False,dimensions=[0,1],weighting_type="uniform",spread=1,pixels=[7,7]):
    """From a graph dataset, generate a list of persistence images (flattened),
    and the associated graph labels

    Parameters
    ----------
    dataset_path:
       Path to the folder corresponding to a dataset. (Ex: Datasets/preprocessed/MUTAG)
    filtration:
        Method to compute the filtration. Must be degree, jaccard or ricci.
    extended:
        If True, use extended persistence. Otherwise, ordinary persistence is used.
    dimensions:
        When extended is False, specifies the dimensions of the components to use.
        Can be [0] for connected components, [1] for cycles or [0,1] for both.
    weighting_type:
        weight applied to the persistence image. Can be 'linear' (as in the original
        paper) or 'uniform' to ignore the weighting.
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

    # Load the graphs and compute the filtration
    graphs=[]
    y=[]
    for f in sorted(os.listdir(dataset_path)):
        graph = ig.Graph.Read_Picklez(os.path.join(dataset_path,f))
        graph = calculate_filtration(graph,method=filtration,attribute_out="f")
        graphs.append(graph)
        y.append(graph["label"])

    # Scale the filtration values, so that they are between 0 and 1.
    # I use a quantile transformation so that the values are uniformly distributed between 0 and 1.
    # Otherwise, if a few nodes have a very high value, then all the other values would be almost identical
    # Ex: if one node has a degree of 1000 and most of the nodes have a degree smaller than 10,
    # then with a linear scaling most nodes would have a value <0.01, and only one pixel of the persistence image would be used.
    graphs = scale_filtration_quantile(graphs,"f")
    
    # Compute persistence diagrams
    # There are 4 diagrams in case of extended persistence. Otherwise, it depends on dimensions.
    persistence_diagrams = [[] , [] ,[] , []] if extended else [[] for dim in dimensions]
    for graph in graphs:
        if extended:
            diagrams = compute_extended_persistence_diagrams(graph,attribute="f")
            for i in range(4):
                if i%2==1: # reverse(birth/death) for diagrams where the death occurs before the birth
                    diagram = [(v,u) for (u,v) in diagrams[i]]
                else:
                    diagram = diagrams[i]
                persistence_diagrams[i].append(diagram)
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
        pim = persim.PersImage(spread=spread, pixels=pixels, verbose=False,weighting_type=weighting_type)
        images.append(pim.transform(dgs))

    # Flatten and concatenate all persistence images
    flattened_images = [np.concatenate([images[ind][i].flatten() for ind in range(len(persistence_diagrams))]) for i in range(len(images[0]))]
    
    return flattened_images,y
