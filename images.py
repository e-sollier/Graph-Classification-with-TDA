import os
import igraph as ig
from pyper.persistent_homology import calculate_persistence_diagrams
from filtrations import calculate_degree_filtration
import persim
import matplotlib.pyplot as plt
import numpy as np


def generate_img_dataset(dataset,spread=1,pixels=[10,10]):
    """From a graph dataset, generate a list of persistence images (flattened), and the associated graph labels"""
    path = "Datasets/preprocessed/"+dataset+"/"
    y=[]
    images=[]
    for f in sorted(os.listdir(path)):
        graph = ig.Graph.Read_Picklez(path+f)
        y.append(graph["label"])
        graph = calculate_degree_filtration(graph,attribute_out='f')
        pd_0, pd_1 = calculate_persistence_diagrams(graph,vertex_attribute='f', edge_attribute='f',order="sublevel")
        pim = persim.PersImage(spread=spread, pixels=pixels, verbose=False)
        img = pim.transform(pd_0._pairs)
        images.append(img.flatten())
    return images,y