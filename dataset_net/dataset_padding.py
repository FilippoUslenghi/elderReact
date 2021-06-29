import os
import pandas as pd
import numpy as np


def find_max_len_df(base_dir):
    """
    Returns the len (number of rows) of the longest dataframe is that specific dataset
    """

    return max([len(pd.read_csv(os.path.join(base_dir, csv))) for csv in os.listdir(base_dir)])


DATASETS = ['train', 'dev', 'test']

max_len_df = 0
for dataset in DATASETS:
    
    len_df = find_max_len_df(os.path.join(dataset, 'delaunay_pose'))
    if  len_df > max_len_df:
        max_len_df = len_df



n_rows = max_len_df
n_cols = 114
print(f'Maximum length: {max_len_df}')
for dataset in DATASETS:

    base_dir = os.path.join(dataset, 'delaunay_pose')
    out_dir = os.path.join(dataset, 'delaunay_pose_padded')
    columns = pd.read_csv(os.path.join(base_dir, os.listdir(base_dir)[0])).columns
    
    for csv in os.listdir(base_dir):

        df = pd.read_csv(os.path.join(base_dir, csv))
        df_values = df.values
        padded_matrix = np.zeros(shape=(n_rows, n_cols))
        padded_matrix[:df_values.shape[0],:df_values.shape[1]] = df_values
        new_df = pd.DataFrame(data=padded_matrix, columns=columns)
        new_df.to_csv(os.path.join(out_dir, csv), index=False)
