import igraph as ig
import numpy as np
import networkx
from scipy.linalg import eigh
from sklearn.preprocessing import QuantileTransformer
from GraphRicciCurvature.OllivierRicci import OllivierRicci

def calculate_filtration(graph, method="degree", useNodeWeight=True, attribute_out='f'):
    """Calculate a filtration for the graph, using the specified method.

    Parameters
    ----------
    graph:
        Input graph
    method:
        Method to compute the filtration. Must be degree, jaccard, ricci, node_betweenness, edge_betweenness or hks.
    useNodeWeight:
        For filtrations based on edges (jaccard, ricci, edgeBetweenness), if True, will set the weight of each node to 
        the minimum weight of its edges. Otherwise, the weight of each node is set to 0.
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
        return calculate_jaccard_filtration(graph,attribute_out=attribute_out,useNodeWeight=useNodeWeight)
    elif method=="ricci":
        return calculate_ricci_filtration(graph,attribute_out=attribute_out,useNodeWeight=useNodeWeight)
    elif method=="nodeBetweenness":
        return calculate_nodeBetweenness_filtration(graph,attribute_out=attribute_out,cutoff=10)
    elif method=="edgeBetweenness":
        return calculate_edgeBetweenness_filtration(graph,attribute_out=attribute_out,cutoff=10,useNodeWeight=useNodeWeight)
    elif method=="hks":
        return calculate_hks_filtration(graph,time=10,attribute_out=attribute_out)
    else:
        raise ValueError("Unrecognized filtration method. Must be one of the following: degree, jaccard, ricci, nodeBetweenness, edgeBetweenness, or hks.")

def scale_filtration_linear(graphs,attribute="f"):
    """
    Apply the transformation x -> (x-min)/(max-min) to the filtration values of the graphs,
    so that they are all between 0 and 1.
    
    Parameters
    ----------
    graphs:
        A list of graphs
    attribute:
        Attribute where the value for the filtration is stored
    """
    min_nodes = np.min([np.min(graph.vs[attribute]) for graph in graphs])
    max_nodes = np.max([np.max(graph.vs[attribute]) for graph in graphs])
    min_edges = np.min([np.min(graph.es[attribute]) for graph in graphs])
    max_edges = np.max([np.max(graph.es[attribute]) for graph in graphs])

    scaled_graphs=[]
    for graph in graphs:
        graph = ig.Graph.copy(graph)
        if max_nodes>min_nodes:
            graph.vs["f"] = [(x-min_nodes)/(max_nodes-min_nodes) for x in graph.vs[attribute]]
        if max_edges>min_edges:
            graph.es["f"] = [(x-min_edges)/(max_edges-min_edges) for x in graph.es[attribute]]
        scaled_graphs.append(graph)
    return scaled_graphs

def scale_filtration_quantile(graphs,attribute="f"):
    """
    Scale the filtration values of the graphs, so that they are uniformly distributed between 0 and 1.
    
    Parameters
    ----------
    graphs:
        A list of graphs
    attribute:
        Attribute where the value for the filtration is stored
    """
    values = []
    for graph in graphs:
        values+= graph.vs[attribute]
    values = np.reshape(values,(-1,1))
    scaler = QuantileTransformer()
    scaler.fit(values)

    scaled_graphs=[]
    for graph in graphs:
        graph = ig.Graph.copy(graph)
        node_values = graph.vs[attribute]
        node_values = np.reshape(node_values,(-1,1))
        scaled_node_values = scaler.transform(node_values)
        graph.vs[attribute] = list(np.reshape(scaled_node_values,(scaled_node_values.shape[0])))
        
        edge_weights = []
        for edge in graph.es:
            a = graph.vs[edge.source][attribute]
            b = graph.vs[edge.target][attribute]
            edge_weights.append(max(a,b))
        graph.es[attribute] = edge_weights
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
    useNodeWeight=True,
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
    node_weights = [1] * len(graph.vs) if useNodeWeight else [0] * len(graph.vs)
    for edge in graph.es:
        u, v = edge.source, edge.target
        inter = len(neighbours[u].intersection(neighbours[v]))
        union = len(neighbours[u].union(neighbours[v]))
        weight = 1- inter/union
        edge_weights.append(weight)
        if useNodeWeight:
            node_weights[u] = min(node_weights[u],weight)
            node_weights[v] = min(node_weights[v],weight)
    graph.es[attribute_out] = edge_weights
    graph.vs[attribute_out] = node_weights
    return graph


def calculate_ricci_filtration(
    graph,
    useNodeWeight=True,
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
    node_weights = [10] * len(graph.vs)
    for edge in graph.es:
        u = edge.source
        v = edge.target
        edge_weights.append(orc.G[u][v]["ricciCurvature"])
        node_weights[u] = min(node_weights[u],orc.G[u][v]["ricciCurvature"])
        node_weights[v] = min(node_weights[v],orc.G[u][v]["ricciCurvature"])
    graph.es[attribute_out] = edge_weights
    if useNodeWeight:
        graph.vs[attribute_out] = node_weights
    else:
        graph.vs[attribute_out] = np.min(edge_weights)

    return graph

def calculate_nodeBetweenness_filtration(
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

def calculate_edgeBetweenness_filtration(
    graph,
    useNodeWeight=True,
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
    node_weights = [np.max(graph.es[attribute_out])] * len(graph.vs)
    for edge in graph.es:
        u = edge.source
        v = edge.target
        node_weights[u] = min(node_weights[u] , edge[attribute_out]) 
        node_weights[v] = min(node_weights[v] , edge[attribute_out]) 
    if useNodeWeight:
        graph.vs[attribute_out] = node_weights
    else:
        graph.vs[attribute_out] = 0
    return graph


def calculate_hks_filtration(
    graph,
    time,
    attribute_out="f",
):
    """Calculate a filtration based on the Heat Kernel Signature (used in PersLay).

    Parameters
    ----------
    graph:
        Input graph
    time:  
        parameter for the Heat Kernel Signature
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
    L = graph.laplacian(normalized = True)
    eigenvals, eigenvectors = eigh(L)
    graph.vs[attribute_out] = np.square(eigenvectors).dot(np.diag(np.exp(-time * eigenvals))).sum(axis=1)

    edge_weights=[]
    for edge in graph.es:
        b1 = graph.vs[edge.source][attribute_out]
        b2 = graph.vs[edge.target][attribute_out]
        edge_weights.append(max(b1,b2))
    graph.es[attribute_out] = edge_weights

    return graph

def add_random_noise(graphs,attribute="f"):
    """
    Add small random noise to the filtration values. This is to ensure that
    all vertices will results in a tuple in the persistence diagram.
    
    Parameters
    ----------
    graphs:
        A list of graphs
    attribute:
        Attribute where the value for the filtration is stored
    """
    modified_graphs=[]
    for graph in graphs:
        graph = ig.Graph.copy(graph)
        # Add random noise
        node_values = graph.vs[attribute]
        noise = np.random.rand(len(node_values)) /1000
        graph.vs[attribute] = [node_values[i] + noise[i] for i in range(len(node_values))]
        
        # Adjust edge weights accordingly
        edge_weights = []
        for edge in graph.es:
            a = graph.vs[edge.source][attribute]
            b = graph.vs[edge.target][attribute]
            edge_weights.append(max(a,b))
        graph.es[attribute] = edge_weights
        modified_graphs.append(graph)
    return modified_graphs