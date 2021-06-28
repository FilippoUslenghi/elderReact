import os
import pandas as pd
import numpy as np

datasets = ['train', 'dev', 'test']
for dataset in datasets:
    
    base_dir = os.path.join(dataset, 'delaunay_pose')
    max_df = pd.DataFrame()
    for csv in os.listdir(base_dir):

        df = pd.read_csv(os.path.join(base_dir, csv))
        print(len(df), csv)
        break
    break
