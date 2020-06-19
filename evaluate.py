import os
import argparse
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from images import generate_img_dataset



def evaluate(dataset,filtration,extended,dimensions,output):
    scoring_metric = "accuracy"
    param_grid = [{ "n_estimators": [ 100], "max_depth":[5, 10, 30]}]

    images,y = generate_img_dataset(dataset=dataset,filtration=filtration,extended = extended, dimensions = dimensions,
                                    spread = 0.2, pixels=[7,7])
    #Perform 10 times a 10-fold cross-validation, with an inner 5-fold CV for hyperparameter tuning
    mean_accuracies=[]
    for i in range(10):
        clf = GridSearchCV(RandomForestClassifier(),param_grid=param_grid,scoring = scoring_metric,n_jobs=-1,cv=5)
        scores = cross_val_score(clf,X=images,y=y,cv=10)
        mean_accuracies.append(np.mean(scores))
    result = "{:.1f}\u00B1{:.1f}".format(np.mean(mean_accuracies)*100,
                                                        np.std(mean_accuracies)*100)

    filename = "{}_{}_".format(dataset,filtration)
    if extended:
        filename+="ext.txt"
    else:
        filename+=str(dimensions)+".txt"


    if not os.path.exists(output):
        os.makedirs(output)
    with open(os.path.join(output,filename),"w") as f:
        f.write(result)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', required=True,type=str, help='Path to the dataset on which to perform the evaluation')
    parser.add_argument('-f', '--filtration', required=True,type=str, help='Filtration to use')
    parser.add_argument('--dim', required=True,type=int, help='Number of dimensions to use. If 4, use extended persistence.')
    parser.add_argument('-o', '--output',required=True,type=str, help='Output folder')

    args = parser.parse_args()
    extended = args.dim ==4
    if args.dim==2:
        dimensions = [0,1]
    else:
        dimensions = [0]
    evaluate(dataset = args.dataset, filtration = args.filtration, extended = extended, dimensions = dimensions, output = args.output)
    
