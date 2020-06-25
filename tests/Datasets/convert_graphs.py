# Used to convert datasets of graphs downloaded from https://chrsmrrs.github.io/datasets/docs/datasets/)
# to igraph graphs.

# Can either be used to convert one dataset, or all the datasets in a directory
 
# To convert one dataset:
# python convert_graphs.py -i raw -o preprocessed 
# where 'raw' is a directory containing the dataset to process
# and 'preprocessed' is a directory where the processed graphs will be stored

# To convert several datasets:
# python convert_graphs.py -i raw -o preprocessed --full
# Where 'raw' is a directory containing the datasets to process
# and 'preprocessed' is a directory where the processed datasets will be stored.



import os
import argparse
import numpy as np
import igraph as ig


def convert_graphs(inputDir,outputDir):
    """
    Converts a graph benchmark dataset (downloaded from https://chrsmrrs.github.io/datasets/docs/datasets/)
    to igraph graphs, and stores each graph as a single file.
    This function is designed to be memory efficient: it only loads one graph at a time.
    Parameters
    ----------
    inputDir:
        path to the directory containing the dataset. It must contain a graph indicator file,
        an edge file and a graph label file, and it can optionnaly contain labels and attributes
        for nodes and edges.
    outputDir:
        path to the directory where the converted graphs will be stored.
    """
    dataset_name = os.path.basename(inputDir)

    #Open files. Only the graph indicator file, the edge file and the graph label file are compulsory.
    graph_indicator_file = open(os.path.join(inputDir,dataset_name+"_graph_indicator.txt"))
    edges_file = open(os.path.join(inputDir,dataset_name+"_A.txt"))
    n_graphs = len(open(os.path.join(inputDir,dataset_name+"_graph_labels.txt")).readlines())
    n_digits = int(np.ceil(np.log10(n_graphs))) #Number of digits to use for the name of the file of each graph
    graph_label_file = open(os.path.join(inputDir,dataset_name+"_graph_labels.txt"))

    node_labels_path = os.path.join(inputDir,dataset_name+"_node_labels.txt")
    have_node_labels = os.path.exists(node_labels_path)
    if have_node_labels:
        node_labels_file = open(node_labels_path)
        node_labels=[]

    node_attributes_path = os.path.join(inputDir,dataset_name+"_node_attributes.txt")
    have_node_attributes = os.path.exists(node_attributes_path)
    if have_node_attributes:
        node_attributes_file = open(node_attributes_path)
        node_attributes=[]
    
    edge_labels_path = os.path.join(inputDir,dataset_name+"_edge_labels.txt")
    have_edge_labels = os.path.exists(edge_labels_path)
    if have_edge_labels:
        edge_labels_file = open(edge_labels_path)
        edge_labels=[]

    edge_attributes_path = os.path.join(inputDir,dataset_name+"_edge_attributes.txt")
    have_edge_attributes = os.path.exists(edge_attributes_path)
    if have_edge_attributes:
        edge_attributes_file = open(edge_attributes_path)
        edge_attributes=[]

    
    
    os.makedirs(outputDir, exist_ok=True)
    current_graph_id=1
    vertex_count=0 #Number of vertices that have been parsed
    offset=1 # ID of the first vertex in the current graph
    edges=[]
    edges_set = set() #used to avoid inserting the same edge twice
    reached_end = False

    #Parse vertices one by one
    while not reached_end:
        graph_id=next(graph_indicator_file,"")
        if graph_id=="":
            reached_end=True
        vertex_count+=1
        if graph_id=="" or int(graph_id)!=current_graph_id:
            # We have reached the end of the current graph.
            # This means that the vertices in the current graph have indices < vertex_count
            # We can now find all edges corresponding to this graph
            # We go through all of the edges, until we reach the end of the file
            # or until we find an edge whose vertices belong to the next graph.
            found_all_edges=False
            while not found_all_edges:
                edge = next(edges_file,"")
                if edge =="": #if we reached the end of the file
                    found_all_edges=True
                else:
                    edge = edge.split(",")
                    edge = (int(edge[0]),int(edge[1]))
                    if have_edge_labels:
                        edge_label = float(next(edge_labels_file))
                    if have_edge_attributes:
                        attr = next(edge_attributes_file).split(",")
                        edge_attribute = np.array([float(a) for a in attr])
                    # The graphs are undirected, so we include each edge only once.
                    # This is achieved here by keeping only edges where the source is smaller than the target
                    # Also make sure that each edge is inserted only once 
                    if edge[0]>edge[1] or (edge[0]-offset,edge[1]-offset) in edges_set:
                        continue
                    if edge[0]<vertex_count and edge[1]<vertex_count:
                        # If the edge is part of the current graph, we add it and keep looking for edges
                        edges.append((edge[0]-offset,edge[1]-offset)) #Make vertex IDs start at 0 in each graph
                        edges_set.add((edge[0]-offset,edge[1]-offset))
                        if have_edge_labels:
                            edge_labels.append(edge_label)
                        if have_edge_attributes:
                            edge_attributes.append(edge_attribute)
                    else:
                        # If the edge belongs to the next graph, we can build the current graph with all the edges that were found
                        found_all_edges = True
                    
            # Once all edges have been found, the graph can be built
            G = ig.Graph(edges)
            G["label"] = int(next(graph_label_file))
            if have_node_labels:
                G.vs["label"] = node_labels
                node_labels=[]
            if have_node_attributes:
                G.vs["attribute"] = node_attributes
                node_attributes=[]
            if have_edge_labels:
                G.es["label"] = edge_labels
                edge_labels = [edge_label]
            if have_edge_attributes:
                G.es["attribute"] = edge_attributes
                edge_attributes=[]

            filename = f'{(current_graph_id-1):0{n_digits}d}.pickle'
            filename = os.path.join(outputDir, filename)
            G.write_picklez(filename)
            offset = vertex_count
            current_graph_id+=1
            if edge!="":
                edges = [(edge[0]-offset,edge[1]-offset)]
                edges_set = {(edge[0]-offset,edge[1]-offset)}

        if have_node_labels and not reached_end:
            node_labels.append(float(next(node_labels_file)))
        if have_node_attributes and not reached_end:
            attr = next(node_attributes_file).split(",")
            node_attributes.append(np.array([float(a) for a in attr]))

    graph_indicator_file.close()
    edges_file.close()
    if have_node_labels:
        node_labels_file.close()
    if have_edge_labels:
        edge_labels_file.close()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True,type=str, help='Input directory')
    parser.add_argument('-o', '--output',required=True,type=str, help='Output directory')
    parser.add_argument('--full', action='store_true', 
    help="If specified, will convert all the datasets in the directory. Otherwise, will only convert one dataset")

    args = parser.parse_args()
    if args.full:
        for dataset in sorted(os.listdir(args.input)):
            if not os.path.exists(os.path.join(args.output,dataset)):
                print("Converting dataset {}...".format(dataset))
                convert_graphs(os.path.join(args.input,dataset),os.path.join(args.output,dataset))
            else:
                print("Dataset {} was already converted. Nothing was done.".format(dataset))
    else:
        convert_graphs(args.input,args.output)