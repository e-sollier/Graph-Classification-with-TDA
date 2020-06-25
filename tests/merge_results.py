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
            results[dataset][method] = result_file.read()
    
    df = pd.DataFrame(results)
    df.index.rename("Method",inplace=True)

    #Compute average accuracy for each method
    averages = []
    for method in df.index:
        values = []
        for dataset in df.columns:
            accuracy = str(df.loc[method,dataset]).split("\u00B1")[0]
            values.append(float(accuracy))
        averages.append(np.mean(values))
    df["Average"] = averages

    df.to_csv(args.output)