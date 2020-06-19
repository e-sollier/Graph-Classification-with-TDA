import gudhi as gd
import igraph as ig
import os
from filtrations import calculate_filtration
from pyper.persistent_homology import calculate_persistence_diagrams


def compute_extended_persistence_diagrams(graph,attribute):
    st = gd.SimplexTree()
    for u in graph.vs:
        st.insert([u.index],filtration = u[attribute])
    for e in graph.es:
        st.insert([e.source,e.target],filtration = e[attribute])
    st.extend_filtration()
    dgs= st.extended_persistence()
    return [[t[1] for t in dg] for dg in dgs]


