import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split


def get_y(group, pose):

    annotations_path = os.path.join(
        'dataset_net', 'Annotations', f'{group}_labels.txt')
    annotations_df = pd.read_csv(
        annotations_path, delim_whitespace=True, header=None)
    all_videos = annotations_df[0]

    y_videos = os.listdir(os.path.join(
        'dataset_net', 'Features', group, f'delaunay_pose_{pose}'))
    y_videos = [y.replace('csv', 'mp4') for y in y_videos]

    boolean_map = [video in y_videos for video in all_videos]
    valence = annotations_df[boolean_map][8] # select the valence column
    y_labels = [int(value>=4) for value in valence]
    
    return y_labels


def read_data(group, pose):
    X = []
    y = []

    X_dir = os.path.join('dataset_net', 'Features', group, f'delaunay_pose_{pose}')
    for csv in sorted(os.listdir(X_dir)):
        df = pd.read_csv(os.path.join(X_dir, csv))
        df = df.drop(columns=['frame','yaw'])
        X.append(df.values)

    y = get_y(group, pose)
    
    return X, y

pose = 'tilted' # tilted or frontal

X, y = read_data('train', pose)
x_val, y_val = read_data('dev', pose)
x_test, y_test = read_data('test', pose)

# split validation in order to increase train and test
new_X, new_x_test, new_y, new_y_test = train_test_split(x_val, y_val)

X += new_X
x_test += new_x_test
y += new_y
y_test += new_y_test

# converting to numpy.ndarray
X, x_test, y, y_test = np.asarray(X, dtype=np.ndarray), np.asarray(x_test, dtype=np.ndarray), np.asarray(y), np.asarray(y_test)

# feature normalization
X = [scale(sample, axis=1) for sample in X]
x_test = [scale(sample, axis=1) for sample in x_test]

# subsamplig???
