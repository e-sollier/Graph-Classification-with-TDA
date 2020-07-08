import os
import numpy as np
import matplotlib.pyplot as plt
import igraph as ig
import numpy as np
from scipy import stats
from sklearn.preprocessing import QuantileTransformer
from sklearn.neighbors import KernelDensity

from pyper.persistent_homology import calculate_persistence_diagrams

from GCTDA.extendedPersistence import compute_extended_persistence_diagrams
from GCTDA.filtrations import calculate_filtration, scale_filtration_quantile, add_random_noise

def generate_density_features(graphs,method,res=10,spread=0.2):
    """From a graph dataset, generate a list of density features.

    Parameters
    ----------
    dataset_path:
       Path to the folder corresponding to a dataset. (Ex: Datasets/preprocessed/MUTAG)
    method:
        List of kernel densities separated by "+".
        Each density can be either 1D or 2D. If 2D, the two functions must be separated by "*"
    res:
        Resolution, ie number of points returned for each kernel density.
    spread:
        Standard deviation of gaussian used to estimate the density
    """
    
    # Method is a list of method names, separated by +. 
    # The densities of all the methods will be concatenated in the final feature vector.
    # Each density is either 1D or 2D. It is 2D if two function names are separated by a *.
    methods = method.split("+")
    densities_list = [] #each element in this list correspond to one method
    for method in methods:
        if not "*" in method: #1D
            values_list = calculate_values_graphs(graphs,method)
            densities = []
            for values in values_list:
                if len(values)==0:
                    densities.append(np.array([0]*res))
                elif len(values)==1:
                    densities.append([stats.norm.cdf((i+1)/res,values[0],spread) - stats.norm.cdf(i/res,values[0],spread) for i in range(res)])
                else:
                    kernel = stats.gaussian_kde(values,spread)
                    bounds = [(x/res,(x+1)/res) for x in range(res)]
                    density = [kernel.integrate_box_1d(x,y)*len(values) for (x,y) in bounds]
                    densities.append(density)
            densities_list.append(densities)
        else: #2D
            method1,method2 = method.split("*")
            values_list1 = calculate_values_graphs(graphs,method1)
            values_list2 = calculate_values_graphs(graphs,method2)
            
            densities=[]
            for i in range(len(values_list1)):
                #values2D = np.reshape([values_list1[i],values_list2[i]],(-1,2))
                values2D = np.array([values_list1[i],values_list2[i]])
                if len(values_list1[i])==0:
                    densities.append([0]*(res*res))
                elif len(values_list1[i])==1 or same_values(values2D):
                    density = []
                    for i in range(res):
                        for j in range(res):
                            p1 = stats.norm.cdf((i+1)/res,values_list1[i][0],spread) - stats.norm.cdf(i/res,values_list1[i][0],spread)
                            p2 = stats.norm.cdf((j+1)/res,values_list2[i][0],spread) - stats.norm.cdf(j/res,values_list2[i][0],spread)
                            density.append(p1*p2 * values2D.shape[1])
                    densities.append(density)
                else:
                    kernel = stats.gaussian_kde(values2D,spread)
                    density=[]
                    for i in range(res):
                        for j in range(res):
                            integral = kernel.integrate_box(low_bounds = [i/res,j/res], high_bounds =[(i+1)/res,(j+1)/res]) * values2D.shape[1]
                            density.append(integral)
                    densities.append(density)
            densities_list.append(densities)
    
    features_list = [np.concatenate([densities_list[dens_index][i] for dens_index in range(len(densities_list))]) for i in range(len(graphs))]
    return features_list

def calculate_values_graphs(graphs,method):
    """
    For each graph in graphs, return a list of features defined by method
    """
    values_list = []
    if method in ["degree","hks","nodeBetweenness"]:
        for graph in graphs:
            graph = calculate_filtration(graph,method=method,attribute_out="f")
            values_list.append(graph.vs["f"])
    elif method in ["jaccard","ricci","edgeBetweenness"]:
        for graph in graphs:
            graph = calculate_filtration(graph,method=method,attribute_out="f")
            values_list.append(graph.es["f"])
    elif method=="degreeNeighbors":
        for graph in graphs:
            values= calculate_degreeNeighbors(graph,method="mean")
            values_list.append(values)
    elif method[:6]=="birth.":
        values_list = calculate_persistence(graphs,filtration=method[6:],dim=0,ind=0)
    elif method[:3]=="cc.":
        values_list = calculate_persistence(graphs,filtration=method[3:],dim=0)
    elif method[:7]=="cycles.":
        values_list = calculate_persistence(graphs,filtration=method[7:],dim=1)
    values_list = scale_quantile(values_list)

    #Add random noise to make sure that all values are not identical
    values_list_noise = []
    for values in values_list:
        noise = np.random.rand(len(values)) /1000
        values_list_noise.append([values[i] + noise[i] for i in range(len(values))])
    return values_list_noise

def scale_quantile(values_list):
    """
    Scales the values, so that they are uniformly distributed between 0 and 1.
    The input is a list of lists (one list for each graph). 
    """
    values_complete = []
    for values in values_list:
        values_complete+= values
    values_complete = np.reshape(values_complete,(-1,1))
    scaler = QuantileTransformer()
    scaler.fit(values_complete)

    scaled_values_list=[]
    for values in values_list:
        if len(values)>0:
            values = np.reshape(values,(-1,1))
            scaled_values = scaler.transform(values)
            scaled_values_list.append(list(np.reshape(scaled_values,(scaled_values.shape[0]))))
        else:
              scaled_values_list.append([])
    return scaled_values_list

def calculate_degreeNeighbors(graph,method="mean"):
    values = []
    for u in graph.vs:
        degrees=[]
        if len(u.neighbors())==0:
            values.append(0)
        else:
            for v in u.neighbors():
                degrees.append(v.degree())
            if method=="mean":
                values.append(np.mean(degrees))
            elif method=="max":
                values.append(np.max(degrees))
            elif method=="min":
                values.append(np.min(degrees))
    return values

def calculate_persistence(graphs,filtration="edgeBetweenness",dim=0,ind=None):
    """
    Report the death times of connected components (dim=0) or the birth times of cycles (dim=1)
    when using the specified filtration.
    """
    if ind is None:
        ind = 1 if dim==0 else 0 # Usually, death for connected components and birth for cycles
    diagrams=[]
    for graph in graphs:
        graph = calculate_filtration(graph,method=filtration,useNodeWeight=False, attribute_out="f")
        pd_0, pd_1 = calculate_persistence_diagrams(graph,vertex_attribute='f', edge_attribute='f')
        if dim==0:
            persistences = [x[ind] for x in pd_0._pairs]
        elif dim==1:
            persistences = [x[ind] for x in pd_1._pairs]
        else:
            raise ValueError("Incorrect dimension. Must be 0 or 1.")
        diagrams.append(persistences)
    return diagrams

def same_values(values2D):
    """
    Returns True if all the pairs are almost identical.
    """
    max_diff0=0
    max_diff1=0
    for i in range(values2D.shape[1]):
        diff0 = (values2D[0,0] - values2D[0,i])**2
        diff1 = (values2D[1,0] - values2D[1,i])**2
        max_diff0 = max(max_diff0,diff0)
        max_diff1 = max(max_diff1,diff1)
    return max_diff0<0.05 or max_diff1<0.05




