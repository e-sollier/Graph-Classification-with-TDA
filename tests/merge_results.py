# Usage: python merge_results.py results_dir -o results.csv
# This script collects results in the directory results_dir
# (where each file contains the accuracy for one dataset with one method)
# and assembles them into one csv file.


import os
import argparse
import numpy as np
import pandas as pd


if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path',type=str, help='Path to the directory containing partial results')
    parser.add_argument('--latex', action='store_true', 
    help="If specified, will format the table so that it can be easily inserted into latex")
    parser.add_argument('-o', '--output',required=True,type=str, help='Output file')
    args = parser.parse_args()

    results = {}
    # Each file contains the accuracy of one method on one dataset
    for f in sorted(os.listdir(args.path)):
        desc = f[:-4] #remove .txt
        desc = desc.split("_")
        dataset = desc[0]
        method = "_".join(desc[1:])
        if not dataset in results:
            results[dataset] = {}
        with open(os.path.join(args.path,f),'r') as result_file:
            results_string = result_file.read()
            if args.latex:
                accuracy,std = results_string.split("\u00B1")
                results_string_latex = "{}$\\pm${}".format(accuracy,std)
                results[dataset][method] = results_string_latex
            else:
                results[dataset][method] = results_string
    
    df = pd.DataFrame(results)
    df.index.rename("Method",inplace=True)

    #Compute average accuracy for each method
    averages = []
    for method in df.index:
        values = []
        for dataset in df.columns:
            if args.latex:
                accuracy = accuracy = str(df.loc[method,dataset]).split("$")[0]
            else:
                accuracy = str(df.loc[method,dataset]).split("\u00B1")[0]
            values.append(float(accuracy))
        averages.append(np.mean(values))
        averages = ["{:.1f}".format(float(x)) for x in averages]
    df["AVERAGE"] = averages

    #Reorder the datasets
    order = ["MUTAG","NCI1","NCI109","COX2","BZR","DHFR","PTC-FM","PTC-FR","PTC-MM","PTC-MR","PROTEINS","DD","COLLAB","IMDB-BINARY","IMDB-MULTI",
            "REDDIT-BINARY","REDDIT-MULTI-5K","REDDIT-MULTI-12K"]
    datasets_ordered = []
    for dataset in order:
        if dataset in df.columns:
            datasets_ordered.append(dataset)
    for dataset in df.columns:
        if not dataset in datasets_ordered:
            datasets_ordered.append(dataset)
    df = df [datasets_ordered]

    # Set datasets to rows and methods to columns (easier to read when there are many datasets)
    df = df.transpose()
    

    df.to_csv(args.output)