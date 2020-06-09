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
        persistence diagrams. Can be either 'sublevel' for a sublevel set
        filtration, or 'superlevel' for a superlevel set filtration.
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
    images=[]
    for f in sorted(os.listdir(path)):
        graph = ig.Graph.Read_Picklez(path+f)
        y.append(graph["label"])
        graph = calculate_filtration(graph,method=filtration,attribute_out='f')
        pd_0, pd_1 = calculate_persistence_diagrams(graph,vertex_attribute='f', edge_attribute='f',order=order)
        pim = persim.PersImage(spread=spread, pixels=pixels, verbose=False)
        img = pim.transform(pd_0._pairs)
        images.append(img.flatten())
    return images,y