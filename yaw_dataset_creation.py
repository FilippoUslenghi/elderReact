import os
import pandas as pd
from collections import Counter

def find_top_angles(n):
    """
    Finds the n most frequent yaw angles
    """

    groups = ['train', 'dev', 'test']
    values = []
    for group in groups:
        base_dir = os.path.join('dataset_net', 'Features', group, 'delaunay_pose')
        
        for csv in os.listdir(base_dir):
            df = pd.read_csv(os.path.join(base_dir, csv))
            values.extend(df['yaw'])
            
    yaw = {'angle': values}
    frequency = Counter([round(value, 2) for value in yaw['angle']])
    sorted_frequency = [(k, v) for k, v in sorted(frequency.items(), key=lambda item: item[1], reverse=True)]
    top_angles = [tup[0] for tup in sorted_frequency[:n]]
    return top_angles

TOP_ANGLES = find_top_angles(5)
groups = ['train', 'dev', 'test']
for group in groups:
    base_dir = os.path.join('dataset_net', 'Features', group, 'delaunay_pose')

    for csv in os.listdir(base_dir):
        df = pd.read_csv(os.path.join(base_dir,csv))
        new_df = df[df['yaw'].round(2).isin(TOP_ANGLES)]
        if len(new_df)!=0:
            new_df.to_csv(os.path.join('dataset_net', 'Features', group, 'delaunay_pose_tilted', csv), index=False)
