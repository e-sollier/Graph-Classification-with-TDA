import igraph as ig


def calculate_degree_filtration(
    graph,
    attribute_out='f'
):
    """Calculate a degree-based filtration for a given graph.
    ----------
    graph:
        Input graph
    attribute_out:
        Specifies the attribute name for storing the result of the
        calculation. This name will pertain to *both* vertices and
        edges.
    Returns
    -------
    Copy of the input graph, with vertex weights and each weights added
    as attributes `attribute_out`, respectively.
    """
    # Let's  make a copy first because we are modifying the graph's
    # attributes in place here.
    graph = ig.Graph.copy(graph)

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