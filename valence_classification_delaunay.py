import os
import math
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, classification_report
from sklearn.model_selection import GridSearchCV

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

pose = 'tilted' # tilted or frontal
X, y = read_data('train', pose)
X_val, y_val = read_data('dev', pose)
X_test, y_test = read_data('test', pose)

# split validation in order to increase train and test
new_X, new_X_test, new_y, new_y_test = train_test_split(X_val, y_val)

# increase train and test
X += new_X; X_test += new_X_test; y += new_y; y_test += new_y_test

# converting to numpy.ndarray
X, X_test, y, y_test = np.asarray(X, dtype=np.ndarray), np.asarray(X_test, dtype=np.ndarray), np.asarray(y), np.asarray(y_test)

# feature normalization
X = [scale(sample, axis=1) for sample in X]
X_test = [scale(sample, axis=1) for sample in X_test]

# subsamplig
X,  y = subsampling(X, y)

# SVM
num_iter = 100
all_pred = []
all_prob = []
all_f1 = 0

e = -4  # hyperparameters to be tuned
c = math.pow(10, e)
clf = SVC(gamma='scale', C=c)

for i in range(num_iter):
    clf.fit(X, y)
    y_pred = clf.predict(X_test)
    all_pred.append(y_pred)

all_pred = np.asarray(all_pred)
final_pred, _ = stats.mode(all_pred)  # voting
final_pred = final_pred[0]

print("accuracy score is...")
print(accuracy_score(y_test, final_pred))
print("F1 score is...")
print(f1_score(y_test, final_pred))
print("Cohen Kappa score is..")
print(cohen_kappa_score(y_test, final_pred, weights='linear'))