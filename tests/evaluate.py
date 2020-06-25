# This script computes the accuracy of one classification method for one dataset.
# A 10-fold nested cross-validation is repeated 10 times.
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
from GCTDA.images import generate_img_dataset
from GCTDA.baseline import count_nodes, histogram_degrees, laplacian_spectrum, heat_kernel_trace



def evaluate(dataset_path,filtration,extended,dimensions,add_features,weighting_type,classifier,baseline,output):
    scoring_metric = "accuracy"
    dataset_name = os.path.basename(dataset_path)
    if classifier =="RF":
        param_grid = [{ "n_estimators": [ 100], "max_depth":[3, 5, 10, 30]}]
    else:
        param_grid = [{ "kernel": ["linear"], "C":[0.1,1,10]}]


    if baseline=="no": #use persistence images
        X,y = generate_img_dataset(dataset=dataset_path,filtration=filtration,extended = extended, dimensions = dimensions, 
                                    weighting_type=weighting_type, spread = 0.2, pixels=[7,7])
        filename = "{}_{}_{}_{}_{}_{}.txt".format(dataset_name,filtration,dimensions,add_features,weighting_type,classifier)
    else: #use a baseline method
        if baseline == "count":
            X,y=count_nodes(dataset_path)
        elif baseline=="degrees":
            X,y = histogram_degrees(dataset_path)
        elif baseline == "spectrum":
            X,y = laplacian_spectrum(dataset_path)
        elif baseline=="heat_kernel_trace":
            X,y = heat_kernel_trace(dataset_path)
        else:
            raise ValueError("Unrecognized baseline method.")
        filename = "{}_{}.txt".format(dataset_name,baseline)

    #pca = PCA(n_components=20)
    #X = pca.fit_transform(X)

    if add_features: #add degree histogram to the features
        X2,y2 = histogram_degrees(dataset_path)
        X = [np.concatenate([X[i],X2[i]]) for i in range(len(X))]

    #Perform 10 times a 10-fold cross-validation, with an inner 5-fold CV for hyperparameter tuning
    mean_accuracies=[]
    for i in range(10):
        cv = StratifiedKFold(n_splits=10, shuffle=True,random_state=i)
        if classifier=="RF":
            clf = GridSearchCV(RandomForestClassifier(),param_grid=param_grid,scoring = scoring_metric,n_jobs=-1,cv=5)
        else:
            clf = GridSearchCV(SVC(),param_grid=param_grid,scoring = scoring_metric,n_jobs=-1,cv=5)
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
    parser.add_argument('-d', '--dataset', required=True,type=str, help='Path to the dataset on which to perform the evaluation')
    parser.add_argument('-f', '--filtration', type=str, help='Filtration to use')
    parser.add_argument('-w', '--weight',type=str, default = "uniform", help='Weighting to use for the persistence image. Can be linear or uniform.')
    parser.add_argument('--dim', type=int, help='Number of dimensions to use. If 4, use extended persistence.')
    parser.add_argument('--add_features',type =str, default="False", help="If True, will also use the degree histogram as features")
    parser.add_argument('-c', '--classifier',type =str, default="RF", help="Type of classifier to use. Can be RF or SVM.")
    parser.add_argument('-b', '--baseline',type =str, default="no", help="Must be one of the baseline methods, which will be used instead of persistent homology.")
    parser.add_argument('-o', '--output',required=True,type=str, help='Output folder')

    args = parser.parse_args()

    dataset_path = os.path.join("Datasets/preprocessed",args.dataset)

    # Ordinary or extended persistence
    extended = args.dim ==4
    if args.dim==1:
        dimensions=[0]
    elif args.dim==2:
        dimensions = [0,1]
    else:
        dimensions = [4]

    add_features = args.add_features=="True"
    np.random.seed(1)

    evaluate(dataset_path = dataset_path, filtration = args.filtration, extended = extended, dimensions = dimensions,
        add_features = add_features, weighting_type=args.weight, classifier=args.classifier, baseline = args.baseline, output = args.output)
    
