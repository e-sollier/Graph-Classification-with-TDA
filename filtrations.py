import igraph as ig
from scipy.stats import wasserstein_distance

def calculate_filtration(graph, method="degree",attribute_out='f'):
    """Calculate a filtration for the graph, using the specified method.

    Parameters
    ----------
    graph:
        Input graph
    method:
        Method to compute the filtration. Must be degree, jaccard or ricci.
    attribute_out:
        Specifies the attribute name for storing the result of the
        calculation. This name will pertain to *both* vertices and
        edges.
    Returns
    -------
    Copy of the input graph, with vertex weights and edge weights added
    as attributes `attribute_out`, respectively.
    """
    if method=="degree":
        return calculate_degree_filtration(graph, attribute_out=attribute_out)
    elif method=="jaccard":
        return calculate_jaccard_filtration(graph,attribute_out=attribute_out)
    elif method=="ricci":
        return calculate_ricci_filtration(graph,attribute_out=attribute_out)
    else:
        raise ValueError("Unrecognized filtration method. Please use degree, jaccard or ricci.")

def calculate_degree_filtration(
    graph,
    attribute_out='f'
):
    """Calculate a degree-based filtration for a given graph.

    Parameters
    ----------
    graph:
        Input graph
    attribute_out:
        Specifies the attribute name for storing the result of the
        calculation. This name will pertain to *both* vertices and
        edges.
    Returns
    -------
    Copy of the input graph, with vertex weights and edge weights added
    as attributes `attribute_out`, respectively.
    """
    graph = ig.Graph.copy(graph)
    graph.vs[attribute_out] = 0 #This value won't be updated for vertices with degree 0
    edge_weights = []

    for edge in graph.es:

        u, v = edge.source, edge.target
        p = graph.degree(u)
        q = graph.degree(v)
        graph.vs[u][attribute_out] = p
        graph.vs[v][attribute_out] = q
        edge_weights.append(max(p,q))

    graph.es[attribute_out] = edge_weights

    return graph

def calculate_jaccard_filtration(
    graph,
    attribute_out='f'
):
    """Calculate a jaccard-based filtration for a given graph.
    The weight of an edge (u,v) is 1 - |neighbours of u and v| / |neighbors of u or v|

    Parameters
    ----------
    graph:
        Input graph
    attribute_out:
        Specifies the attribute name for storing the result of the
        calculation. This name will pertain to *both* vertices and
        edges.
    Returns
    -------
    Copy of the input graph, with vertex weights and edge weights added
    as attributes `attribute_out`, respectively.
    """
    graph = ig.Graph.copy(graph)
     
    #Compute the set of neighbours of each vertex
    neighbours = [ set() for _ in graph.vs]
    for edge in graph.es:
        u, v = edge.source, edge.target
        neighbours[u].add(v)
        neighbours[v].add(u)

    #Compute the Jaccard index of each edge
    edge_weights = []
    for edge in graph.es:
        u, v = edge.source, edge.target
        inter = len(neighbours[u].intersection(neighbours[v]))
        union = len(neighbours[u].union(neighbours[v]))
        edge_weights.append(1 - inter/union)

    graph.es[attribute_out] = edge_weights
    graph.vs[attribute_out] = 0

    return graph

def calculate_ricci_filtration(
    graph,
    attribute_out='f',
    alpha=0.5
):
    """Calculate a filtration based on Ollivier's Ricci curvature for 
    a given graph (as described by Zhao and Wang, 2019).

    Parameters
    ----------
    graph:
        Input graph
    attribute_out:
        Specifies the attribute name for storing the result of the
        calculation. This name will pertain to *both* vertices and
        edges.
    alpha:  
        Parameter used to compute the Ricci curvature. Was set to 0.5 by Zhao and Wang.
    Returns
    -------
    Copy of the input graph, with vertex weights and edge weights added
    as attributes `attribute_out`, respectively.
    """
    graph = ig.Graph.copy(graph)
     
    #Compute the set of neighbours of each vertex
    neighbours = [ set() for _ in graph.vs]
    for edge in graph.es:
        u, v = edge.source, edge.target
        neighbours[u].add(v)
        neighbours[v].add(u)

    #Compute the probability measure associated with each vertex
    values = []
    weights = []
    for u in range(len(graph.vs)):
        n = len(neighbours[u])
        values.append([u] + list(neighbours[u]))
        weights.append([alpha] + [(1-alpha)/n]*n)

    #Compute Ollivier's Ricci curvature of each edge
    edge_weights = []
    for edge in graph.es:
        u, v = edge.source, edge.target
        d = wasserstein_distance(values[u],values[v],weights[u],weights[v])
        edge_weights.append(d)

    graph.es[attribute_out] = edge_weights
    graph.vs[attribute_out] = 0

    return graph