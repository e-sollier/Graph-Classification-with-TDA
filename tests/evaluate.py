# This script computes the accuracy of one classification method for one dataset.
# A 10-fold nested cross-validation is repeated several times.
# The output is a single file, with a name corresponding to the dataset and the method,
# and the content of the file is the mean accuracy and the std.

# This script is intended to be run multiple times, with different parameters
# and the individual results files can the be aggregated with merge_results.py

import os
import sys
import argparse
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

currentdir = os.path.dirname(os.path.abspath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from GCTDA.datasets import load_dataset
from GCTDA.images import generate_image_features
from GCTDA.kernel_density import generate_density_features
from GCTDA.baseline import generate_baseline_features



def evaluate(dataset_name,directory,filtration,baseline,density,extended,dimensions,weighting_type,res,spread,classifier,output):
    scoring_metric = "accuracy"

    # Can use either a Random Forest or a SVM for classification
    if classifier =="RF":
        param_grid = [{ "n_estimators": [100], "max_depth":[3,5,10,30,50]}]
    else:
        param_grid = [{ "kernel": ["linear"], "C":[0.1,1,10]}]


    # For synthetic datasets, the evaluation is repeated with different seeds for the graph generation
    # to get a better estimate of the accuracy
    nb_dataset_repeats = 10 if ":" in dataset_name else 1

    # For synthetic datasets, the 10-fold nested CV is repeated only 5 times per dataset, because 
    # the evaluation is already repeated on several datasets
    nb_CV_repeats = 5 if ":" in dataset_name else 10 

    mean_accuracies = []
    for dataset_seed in range(nb_dataset_repeats): 
        graphs,y = load_dataset(dataset_name,directory=directory,seed=dataset_seed)

        # X is a list of feature vectors, one for each graph.
        X = [[] for graph in graphs]
        method_names = []
        if filtration!="none": #use persistence images
            X2 = generate_image_features(graphs,filtration=filtration,extended = extended, dimensions = dimensions, 
                                        weighting_type=weighting_type, spread = spread, pixels=[res,res])
            X = [np.concatenate([X[i],X2[i]]) for i in range(len(X))]
            method_names.append("images:{},dim{},{},std{},res{}".format(filtration,"-".join([ str(x) for x in dimensions]),weighting_type,spread,res))
        if baseline!="none": #use a baseline method
            X2 = generate_baseline_features(graphs,baseline)
            X = [np.concatenate([X[i],X2[i]]) for i in range(len(X))]
            method_names.append(baseline)
        if density!="none": #use density estimation
            X2 = generate_density_features(graphs,method=density,spread=spread,res=res)
            X = [np.concatenate([X[i],X2[i]]) for i in range(len(X))]
            method_names.append("density:{},std{},res{}".format(density,spread,res))

        filename = dataset_name + "_" + "_".join(method_names) + "_" + classifier

        """
        if baseline=="no" and density=="no": #use persistence images
            X = generate_image_dataset(graphs,filtration=filtration,extended = extended, dimensions = dimensions, 
                                        weighting_type=weighting_type, spread = spread, pixels=[res,res])
            filename = "{}_{}_{}_{}_{}_{}_spread{}_res{}.txt".format(dataset_name,filtration,"dim"+"-".join([ str(x) for x in dimensions]),add_features,
            weighting_type,classifier,spread,res)
        elif density=="no": #use a baseline method
            X = generate_baseline_dataset(graphs,baseline)
            filename = "{}_{}.txt".format(dataset_name,baseline)
        else: #use density estimation
            X = generate_density_dataset(graphs,method=density,spread=spread,res=10)
            filename = "{}_{}_{}.txt".format(dataset_name,density,spread)


        if add_hist: #add degree histogram to the features
            X2 = histogram_degrees(graphs)
            X = [np.concatenate([X[i],X2[i]]) for i in range(len(X))]"""

        #Perform several times a 10-fold cross-validation, with an inner 5-fold CV for hyperparameter tuning
        for CV_seed in range(nb_CV_repeats):
            cv = StratifiedKFold(n_splits=10, shuffle=True,random_state=CV_seed)
            if classifier=="RF":
                clf = GridSearchCV(RandomForestClassifier(random_state=CV_seed+10),param_grid=param_grid,scoring = scoring_metric,n_jobs=-1,cv=5)
            else:
                clf = GridSearchCV(SVC(random_state=CV_seed+10),param_grid=param_grid,scoring = scoring_metric,n_jobs=-1,cv=5)
            scores = cross_val_score(clf,X=X,y=y,cv=cv)
            mean_accuracies.append(np.mean(scores))

    result = "{:.1f}\u00B1{:.1f}".format(np.mean(mean_accuracies)*100,
                                                        np.std(mean_accuracies)*100)
    # Write the result to a file
    if not os.path.exists(output):
        os.makedirs(output)
    with open(os.path.join(output,filename),"w") as f:
        f.write(result)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', required=True,type=str, help='Name of the dataset on which to perform the evaluation')
    parser.add_argument('--dir',type=str, default = "Datasets/preprocessed/", help='Path to the directory containing the graph datasets')
    parser.add_argument('-f', '--filtration', type=str,default="none", help='Filtration to use')
    parser.add_argument('-b', '--baseline',type =str, default="none", help="Must be one of the baseline methods.")
    parser.add_argument('--density',type =str, default="none", help="Use density estimation instead of persistence images.")
    parser.add_argument('-w', '--weight',type=str, default = "uniform", help='Weighting to use for the persistence image. Can be linear or uniform.')
    parser.add_argument('--dim', type=int, help='Number of dimensions to use. If 4, use extended persistence.')
    parser.add_argument('--res',type =int, default=7, help="Resolution (number of pixels in width and height) for persistence images, or number of bins for density estimation.")
    parser.add_argument('--spread',type =float, default=0.2, help="Spread of the gaussians for persistence images or density estimation.")
    parser.add_argument('-c', '--classifier',type =str, default="RF", help="Type of classifier to use. Can be RF or SVM.")
    parser.add_argument('-o', '--output',required=True,type=str, help='Output folder')

    args = parser.parse_args()

    # Ordinary or extended persistence
    extended = args.dim ==4
    if args.dim==1:
        dimensions=[0]
    elif args.dim==2:
        dimensions = [0,1]
    else:
        dimensions = [4]

    np.random.seed(1)
    evaluate(dataset_name = args.dataset,directory = args.dir, filtration = args.filtration, extended = extended, dimensions = dimensions,
        weighting_type=args.weight, classifier=args.classifier,
        res = args.res, spread =args.spread, baseline = args.baseline, density = args.density, output = args.output)
    
