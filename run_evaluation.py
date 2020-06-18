import argparse
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from images import generate_img_dataset



def run_evaluation(config,output):
    scoring_metric = "accuracy"
    param_grid = [{ "n_estimators": [ 100], "max_depth":[5, 10, 30]}]

    results = {}
    for dataset in config["datasets"]:
        results[dataset] = {}
        for filtration in config["filtrations"]:
            for dim in config["dimensions"]:
                images,y = generate_img_dataset(dataset=dataset,filtration=filtration,order="sublevel",
                                                spread = 0.2, pixels=[10,10],dimensions=dim)
                #Perform 10 times a 10-fold cross-validation, with an inner 5-fold CV for hyperparameter tuning
                mean_accuracies=[]
                for i in range(10):
                    clf = GridSearchCV(RandomForestClassifier(),param_grid=param_grid,scoring = scoring_metric,n_jobs=-1,cv=5)
                    scores = cross_val_score(clf,X=images,y=y,cv=10)
                    mean_accuracies.append(np.mean(scores))
                results[dataset][filtration + str(dim)] = "{:.1f}\u00B1{:.1f}".format(np.mean(mean_accuracies)*100,
                                                                    np.std(mean_accuracies)*100)
    df = pd.DataFrame(results)
    df.to_csv(output)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True,type=str, help='JSON file containing the tests to run')
    parser.add_argument('-o', '--output',required=True,type=str, help='Output file')

    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
    run_evaluation(config,args.output)
    
