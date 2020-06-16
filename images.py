import os
import igraph as ig
from pyper.persistent_homology import calculate_persistence_diagrams
from filtrations import calculate_filtration
import persim
import matplotlib.pyplot as plt
import numpy as np


def generate_img_dataset(dataset,filtration="degree",order="sublevel",spread=1,pixels=[10,10],dimensions=[0]):
    """From a graph dataset, generate a list of persistence images (flattened),
    and the associated graph labels

    Parameters
    ----------
    dataset:
        Name of the folder corresponding to a dataset. Ex: MUTAG
    filtration:
        Method to compute the filtration. Must be degree, jaccard or ricci.
    order:
        Specifies the filtration order that is to be used for calculating
        persistence diagrams. Can be 'sublevel' for a sublevel set
        filtration, 'superlevel' for a superlevel set filtration, or
        'both' for a concatenation of both persistance diagrams.
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
    y=[]
    persistence_diagrams = [[] , []]

    # Compute the persistence diagrams
    for f in sorted(os.listdir(path)):
        graph = ig.Graph.Read_Picklez(path+f)
        y.append(graph["label"])
        orders = ["sublevel","superlevel"] if order=="both" else [order]
        persistence_pairs_0=[]
        persistence_pairs_1=[]
        for o in orders:
            graph = calculate_filtration(graph,method=filtration,order=o,attribute_out='f')
            pd_0, pd_1 = calculate_persistence_diagrams(graph,vertex_attribute='f', edge_attribute='f',order=o)
            persistence_pairs_0+=pd_0._pairs 
            persistence_pairs_1+=pd_1._pairs
        persistence_diagrams[0].append(persistence_pairs_0)
        persistence_diagrams[1].append(persistence_pairs_1)

    # Compute the persistence images
    # It is better to compute all the persistence images at once, because that way the
    # same scaling is applied to all the persistence diagrams.
    images = []
    for dim in dimensions:
        pim = persim.PersImage(spread=spread, pixels=pixels, verbose=False,weighting_type="uniform")
        images.append(pim.transform(persistence_diagrams[dim]))
    flattened_images = [np.concatenate([images[dim][i].flatten() for dim in range(len(dimensions))]) for i in range(len(images[0]))]
    
    return flattened_images,y
