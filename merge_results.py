import os
import argparse
import pandas as pd



if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path',type=str, help='Path to the directory containing partial results')
    parser.add_argument('-o', '--output',required=True,type=str, help='Output file')
    args = parser.parse_args()

    results = {}
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
    df.to_csv(args.output)