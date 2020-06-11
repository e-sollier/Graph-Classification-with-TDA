import os
import igraph as ig
from pyper.persistent_homology import calculate_persistence_diagrams
from filtrations import calculate_filtration
import persim
import matplotlib.pyplot as plt
import numpy as np


def generate_img_dataset(dataset,filtration="degree",order="sublevel",spread=1,pixels=[10,10]):
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
    persistence_diagrams = []

    # Compute the persistence diagrams
    for f in sorted(os.listdir(path)):
        graph = ig.Graph.Read_Picklez(path+f)
        y.append(graph["label"])
        orders = ["sublevel","superlevel"] if order=="both" else [order]
        persistant_pairs=[]
        for o in orders:
            graph = calculate_filtration(graph,method=filtration,order=o,attribute_out='f')
            pd_0, pd_1 = calculate_persistence_diagrams(graph,vertex_attribute='f', edge_attribute='f',order=o)
            persistant_pairs+=pd_0._pairs 
        persistence_diagrams.append(persistant_pairs)

        """
        graph = calculate_filtration(graph,method=filtration,attribute_out='f')
        pd_0, pd_1 = calculate_persistence_diagrams(graph,vertex_attribute='f', edge_attribute='f',order=order)
        persistences = [x[1]-x[0] for x in pd_0._pairs] 
        persistence_diagrams.append(pd_0._pairs)"""

    # Compute the persistence images
    # It is better to compute all the persistence images at once, because that way the
    # same scaling is applied to all the persistence diagrams.
    pim = persim.PersImage(spread=spread, pixels=pixels, verbose=False)
    images = pim.transform(persistence_diagrams)
    flattened_images = [img.flatten() for img in images]
    
    return flattened_images,y
