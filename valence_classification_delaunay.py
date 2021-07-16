import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def get_y(group, pose):
    
    labels_path = os.path.join('dataset_net', 'Annotations', f'{group}_labels.txt')
    labels_df = pd.read_csv(labels_path, delim_whitespace=True, header=None)
    
    y_list = os.listdir(os.path.join('dataset_net', 'Features', group, f'delaunay_pose_{pose}'))
    y_list = [y.replace('csv', 'mp4') for y in y_list]
    
    y_df = labels_df[0]
    print(np.where(labels_df[0] in y_list))

def read_data(group, pose):
    X=[]
    y=[]

    # X_dir = os.path.join('dataset_net', 'Features', group, f'delaunay_pose_{pose}')
    # for csv in os.listdir(X_dir):
    #     df = pd.read_csv(os.path.join(X_dir, csv))
    #     df.drop(columns=['frame','yaw'])
    #     X.append(df.values)

    y = get_y(group, pose)

read_data('test', 'tilted')