import os
import igraph as ig
from pyper.persistent_homology import calculate_persistence_diagrams
from pyper.representations import persistence_image
from filtrations import calculate_degree_filtration
import persim
import matplotlib.pyplot as plt
import numpy as np


graph = ig.Graph.Read_Picklez("Datasets/preprocessed/MUTAG/000.pickle")
graph = calculate_degree_filtration(graph)
pd_0, pd_1 = calculate_persistence_diagrams(graph,vertex_attribute='f', edge_attribute='f',order="sublevel")

pd = np.array([[x[0] for x in pd_0], [x[1] for x in pd_0]])
pim = persistence_image.PersistenceImage(resolution=(10,10))
img = pim.transform([pd_0._pairs])


pim = persim.PersImage(spread=1, pixels=[10,10], verbose=False)
img = pim.transform(pd_1._pairs)
plt.show(img)