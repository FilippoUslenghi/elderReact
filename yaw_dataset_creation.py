import os
import pandas as pd

POSE_DICT = {
    'tilted': [0.32, 0.33, 0.31, 0.3, 0.29],
    'frontal': [0.56, 0.55, 0.57, 0.58, 0.54]
}

pose = ['tilted', 'frontal']
selected_pose = 1
angles = POSE_DICT[pose[selected_pose]]

groups = ['train', 'dev', 'test']
for group in groups:
    base_dir = os.path.join('dataset_net', 'Features', group, 'delaunay_pose')

    for csv in os.listdir(base_dir):
        df = pd.read_csv(os.path.join(base_dir, csv))
        new_df = df[df['yaw'].round(2).isin(angles)]
        if len(new_df) != 0:
            new_df.to_csv(os.path.join('dataset_net', 'Features',
                          group, f'delaunay_pose_{pose[selected_pose]}', csv), index=False)
