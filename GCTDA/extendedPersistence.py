import gudhi as gd
import igraph as ig

def compute_extended_persistence_diagrams(graph,attribute):
    """
    Compute the 4 extended persistence diagrams for a graph, using Gudhi.
    """
    st = gd.SimplexTree()
    for u in graph.vs:
        st.insert([u.index],filtration = u[attribute])
    for e in graph.es:
        st.insert([e.source,e.target],filtration = e[attribute])
    st.extend_filtration()
    dgs= st.extended_persistence()
    return [[t[1] for t in dg] for dg in dgs]


