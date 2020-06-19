import igraph as ig
import numpy as np
import networkx
from scipy.stats import wasserstein_distance
from GraphRicciCurvature.OllivierRicci import OllivierRicci

def calculate_filtration(graph, method="degree",attribute_out='f'):
    """Calculate a filtration for the graph, using the specified method.

    Parameters
    ----------
    graph:
        Input graph
    method:
        Method to compute the filtration. Must be degree, jaccard, riccin node_betweenness or edge_betweenness.
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
    elif method=="node_betweenness":
        return calculate_node_betweenness_filtration(graph,attribute_out=attribute_out)
    elif method=="edge_betweenness":
        return calculate_edge_betweenness_filtration(graph,attribute_out=attribute_out)
    else:
        raise ValueError("Unrecognized filtration method. Must be one of the following: degree, jaccard, ricci, node_betweennes, or edge_betweenness.")

def scale_filtration(graphs,attribute="f",individual=False):
    """
    Apply the transformation x -> (x-min)/(max-min) to the filtration values of the graphs,
    so that they are all between 0 and 1.
    
    Parameters
    ----------
    graphs:
        A list of graphs
    attribute:
        Attribute where the value for the filtration is stored
    individual:
        If True, apply the scaling individually to each graph (so the max and min values correspond
        to the max and min of each graph, as opposed to the max and min across the whole dataset).
        This is not recommended, this is just provided for evaluation purposes.
    """
    if not individual:
        min_nodes = np.min([np.min(graph.vs[attribute]) for graph in graphs])
        max_nodes = np.max([np.max(graph.vs[attribute]) for graph in graphs])
        min_edges = np.min([np.min(graph.es[attribute]) for graph in graphs])
        max_edges = np.max([np.max(graph.es[attribute]) for graph in graphs])

    scaled_graphs=[]
    for graph in graphs:
        graph = ig.Graph.copy(graph)
        if individual:
            min_nodes = np.min(graph.vs[attribute])
            max_nodes = np.max(graph.vs[attribute]) 
            min_edges = np.min(graph.es[attribute]) 
            max_edges = np.max(graph.es[attribute]) 
        if max_nodes>min_nodes:
            graph.vs["f"] = [(x-min_nodes)/(max_nodes-min_nodes) for x in graph.vs[attribute]]
        if max_edges>min_edges:
            graph.es["f"] = [(x-min_edges)/(max_edges-min_edges) for x in graph.es[attribute]]
        scaled_graphs.append(graph)
    return scaled_graphs




def calculate_degree_filtration(
    graph,
    attribute_out='f',
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
    graph.vs[attribute_out] = graph.degree()
    edge_weights = []
    for edge in graph.es:
        u, v = edge.source, edge.target
        p = graph.degree(u)
        q = graph.degree(v)
        edge_weights.append(max(p,q))
    graph.es[attribute_out] = edge_weights
    return graph

def calculate_jaccard_filtration(
    graph,
    attribute_out='f',
):
    """Calculate a jaccard index-based filtration for a given graph.
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
    alpha=0.5,
):
    """Calculate a filtration based on Ollivier's Ricci curvature for 
    a given graph. The computation is done using the library GraphRicciCurvature

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

    #Convert the graph to a networkx graph (so that the GraphRicciCurvature library can be used)
    G = networkx.Graph( [(edge.source,edge.target,{'weight':1}) for edge in graph.es] )
    orc = OllivierRicci(G, alpha=0.5, verbose="INFO")
    res = orc.compute_ricci_curvature()

    edge_weights = []
    for edge in graph.es:
        edge_weights.append(orc.G[edge.source][edge.target]["ricciCurvature"])
    graph.es[attribute_out] = edge_weights
    graph.vs[attribute_out] = min(edge_weights)

    return graph

def calculate_node_betweenness_filtration(
    graph,
    attribute_out='f',
    cutoff=None
):
    """Calculate a filtration based on the betweenness centrality of nodes for 
    a given graph.

    Parameters
    ----------
    graph:
        Input graph
    attribute_out:
        Specifies the attribute name for storing the result of the
        calculation. This name will pertain to *both* vertices and
        edges.
    cutoff:
        For the computation of betweenness centrality, only paths of length
        shorter than this cutoff are considered
    Returns
    -------
    Copy of the input graph, with vertex weights and edge weights added
    as attributes `attribute_out`, respectively.
    """
    graph = ig.Graph.copy(graph)

    graph.vs[attribute_out] = graph.betweenness(cutoff=cutoff)

    edge_weights=[]
    for edge in graph.es:
        b1 = graph.vs[edge.source][attribute_out]
        b2 = graph.vs[edge.target][attribute_out]
        edge_weights.append(max(b1,b2))
    graph.es[attribute_out] = edge_weights

    return graph

def calculate_edge_betweenness_filtration(
    graph,
    attribute_out='f',
    cutoff=None
):
    """Calculate a filtration based on the betweenness centrality of edges for 
    a given graph.

    Parameters
    ----------
    graph:
        Input graph
    attribute_out:
        Specifies the attribute name for storing the result of the
        calculation. This name will pertain to *both* vertices and
        edges.
    cutoff:
        For the computation of betweenness centrality, only paths of length
        shorter than this cutoff are considered
    Returns
    -------
    Copy of the input graph, with vertex weights and edge weights added
    as attributes `attribute_out`, respectively.
    """
    graph = ig.Graph.copy(graph)
    graph.es[attribute_out] = graph.edge_betweenness(cutoff=cutoff)
    graph.vs[attribute_out] = 0
    return graph