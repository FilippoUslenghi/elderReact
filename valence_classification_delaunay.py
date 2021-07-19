import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline


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
    valence = annotations_df[boolean_map][8]  # select the valence column
    y_labels = [int(value >= 4) for value in valence]

    return y_labels


def read_data(group, pose):
    videos = []
    labels = []

    videos_dir = os.path.join('dataset_net', 'Features',
                              group, f'delaunay_pose_{pose}')
    for csv in sorted(os.listdir(videos_dir)):
        df = pd.read_csv(os.path.join(videos_dir, csv))
        df = df.drop(columns=['frame', 'yaw'])
        videos.append(df.mean(axis=0).values)

    labels = get_y(group, pose)

    return videos, labels


def subsampling(X, y):
    samples_needed = 0
    for num in y:
        if num == 0:
            samples_needed += 1

    X_neg = []
    X_pos = []
    for i, val in enumerate(y):
        if val == 0:
            X_neg.append(X[i])
        else:
            X_pos.append(X[i])

    X_resample = resample(X_pos, replace=False, n_samples=samples_needed)

    new_X = X_resample + X_neg

    new_y = []
    for i in range(samples_needed):
        new_y.append(1)
    for i in range(samples_needed):
        new_y.append(0)

    return np.asarray(new_X, dtype=np.ndarray), np.asarray(new_y)


pose = 'tilted'  # tilted or frontal
X, y = read_data('train', pose)
X_val, y_val = read_data('dev', pose)
X_test, y_test = read_data('test', pose)

# split validation in order to increase train and test
new_X, new_X_test, new_y, new_y_test = train_test_split(X_val, y_val)

# increase train and test
X += new_X
X_test += new_X_test
y += new_y
y_test += new_y_test

X, X_test, y, y_test = np.asarray(X, dtype=np.ndarray), np.asarray(
    X_test, dtype=np.ndarray), np.asarray(y), np.asarray(y_test)

# create the pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC())
])

# set params for random search
params = {'classifier__C': stats.expon(scale=100),
          'classifier__gamma': stats.expon(scale=.1),
          'classifier__kernel': ['rbf']
          }


num_iter = 100
all_pred = []

for i in range(num_iter):

    X, y = subsampling(X, y)

    randomsearch = RandomizedSearchCV(
        pipe, params, n_iter=20).fit(X, y) # fit the model

    y_pred = randomsearch.predict(X_test)
    all_pred.append(y_pred)

all_pred = np.asarray(all_pred)
final_pred, _ = stats.mode(all_pred) # voting
final_pred = final_pred[0]

# print(f'Best params: {randomsearch.best_params_}')
print(f"accuracy score is: {accuracy_score(y_test, final_pred)}")
print(
    f"Cohen Kappa score is: {cohen_kappa_score(y_test, final_pred, weights='linear')}")
print("classification report:")
print(classification_report(y_test, final_pred))
