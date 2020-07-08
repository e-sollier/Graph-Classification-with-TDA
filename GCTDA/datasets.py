import os
import numpy as np
import igraph as ig


def load_dataset(dataset_name,directory,seed=0):
    """
    Load a dataset, which can either be synthetic, in which case it will be generated based on the provided random seed,
    or real, in which case the files of the graph must be stored in one directory as pickle files.
    """
    if ":" in dataset_name: #synthetic dataset
        algorithm, rewire_probs = dataset_name.split(":")
        rewire_probs = [float(x) for x in rewire_probs.split("-")]
        if algorithm == "BA-rewire":
            X,y = generate_BA_rewire(400,rewire_probs,random_seed=seed)
        elif algorithm =="watts-strogatz":
            X,y = generate_watts_strogatz(400,rewire_probs,random_seed=seed) 
        else:
            raise ValueError("Unrecognized graph generation method. Must be BA-rewire or watts-strogatz.")
    else: #real dataset
        dataset_path = os.path.join(directory,dataset_name)
        X,y = load_real_dataset(dataset_path)
    return X,y



def load_real_dataset(dataset_path):
    """
    Load a graph dataset. The input graphs are supposed to be stored as pickle files in one directory.
    """
    graphs=[]
    y=[]
    for f in sorted(os.listdir(dataset_path)):
        graph = ig.Graph.Read_Picklez(os.path.join(dataset_path,f))
        graphs.append(graph)
        y.append(graph["label"])
    return graphs, y

def generate_barabasi_albert(nbGraphs,n,m,class_ratio=0.5,random_seed=0):
    """
    Generate a dataset of graphs with the Barabasi Albert model.

    Parameters:
    ----------
    nbGraphs: size of the dataset
    n: number of vertices in each graph. This is a list of 2 elements, corresponding to the n value for each class.
    m: When we add a node, to how many nodes it will be connected. This is a list of 2 elements, corresponding to the m value for each class.
    class_ratio: probability to generate a graph from the first class.
    """
    np.random.seed(random_seed)
    graphs = []
    y=[]
    for i in range(nbGraphs):
        if np.random.rand()<class_ratio:
            y.append(0)
            graphs.append(ig.Graph.Barabasi(n[0],m[0]))
        else:
            y.append(1)
            graphs.append(ig.Graph.Barabasi(n[1],m[1]))
    return graphs, y


def generate_watts_strogatz(nbGraphs,p,random_seed=0):
    """
    Generate a dataset of graphs with the Watts Strogatz model.

    Parameters:
    ----------
    nbGraphs: size of the dataset
    p: rewiring probability for each of the 2 classes
    """
    np.random.seed(random_seed)
    graphs = []
    y=[]
    for i in range(nbGraphs):
        size = np.random.randint(20,150)
        nei = np.random.randint(1,6)
        label = np.random.choice([0,1])
        y.append(label)
        graphs.append(ig.Graph.Watts_Strogatz(dim=1,size=size,nei=nei,p=p[label]))
    return graphs, y

def generate_BA_ER(nbGraphs,random_seed=0):
    """
    Generate a dataset of graphs, all with the same number of vertices and edges, but
    some are generated with the Barabasi-Albert method, while others are generated with the Erdos_Renyi algorithm.

    Parameters:
    ----------
    nbGraphs: size of the dataset
    class_ratio: probability to generate a graph from the first class.
    """
    np.random.seed(random_seed)

    graphs = []
    y=[]
    for i in range(nbGraphs):
        nbVertices = np.random.randint(20,130)
        nbEdges = np.random.randint(nbVertices,nbVertices*6)
        m=round(nbEdges / nbVertices)
        label = np.random.choice([0,1])
        y.append(label)

        if label==0:
            graphs.append(ig.Graph.Barabasi(n=nbVertices,m = m))
        else:
            graphs.append(ig.Graph.Erdos_Renyi(n=nbVertices,m=nbEdges))

    return graphs, y

def generate_BA_rewire(nbGraphs,p,random_seed=0):
    """
    Generate a dataset of graphs, all with the same number of vertices and edges, but
    some are generated with the Barabasi-Albert method, while others are generated with the Erdos_Renyi algorithm.

    Parameters:
    ----------
    nbGraphs: size of the dataset
    p: rewiring probabilities for each of the 2 classes
    """
    np.random.seed(random_seed)

    graphs = []
    y=[]
    for i in range(nbGraphs):
        nbVertices = np.random.randint(20,130)
        nbEdges = np.random.randint(nbVertices,nbVertices*6)
        m=round(nbEdges / nbVertices)
        label = np.random.choice([0,1])
        
        graph = ig.Graph.Barabasi(n=nbVertices,m = m)
        graph.rewire_edges(prob=p[label])

        y.append(label)
        graphs.append(graph)

    return graphs, y