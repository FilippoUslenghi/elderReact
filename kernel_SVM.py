from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import cohen_kappa_score, classification_report, plot_confusion_matrix
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from scipy import stats
import seaborn as sns
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')


def get_y(group, pose, emotion):

    annotations_path = os.path.join(
        'dataset_net', 'Annotations', f'{group}_labels.txt')
    annotations_df = pd.read_csv(
        annotations_path, delim_whitespace=True, header=None)
    all_videos = annotations_df[0]

    pose = '' if pose == 'none' else pose
    y_videos = os.listdir(os.path.join(
        'dataset_net', 'Features', group, f'delaunay_pose_{pose}'))
    y_videos = [y.replace('csv', 'mp4') for y in y_videos]

    boolean_map = [video in y_videos for video in all_videos]

    if emotion == 'anger':
        # select the anger column
        y_labels = annotations_df[boolean_map][1].tolist()

    elif emotion == 'disgust':
        # select the disgust column
        y_labels = annotations_df[boolean_map][2].tolist()

    elif emotion == 'fear':
        # select the fear column
        y_labels = annotations_df[boolean_map][3].tolist()

    elif emotion == 'happiness':
        # select the happiness column
        y_labels = annotations_df[boolean_map][4].tolist()

    elif emotion == 'sadness':
        # select the sadness column
        y_labels = annotations_df[boolean_map][5].tolist()

    elif emotion == 'surprise':
        # select the surprise column
        y_labels = annotations_df[boolean_map][6].tolist()

    elif emotion == 'valence':
        valence = annotations_df[boolean_map][8]  # select the valence column
        y_labels = [int(value >= 4) for value in valence]  # binarization

    return y_labels


def read_data(group, pose, emotion, features):
    videos = []
    labels = []

    pose = '' if pose == 'none' else pose
    if features == 'delaunay':
        videos_dir = os.path.join('dataset_net', 'Features',
                                  group, f'delaunay_pose_{pose}')
    elif features[:2] == 'au':
        videos_dir = os.path.join('dataset_net', 'Features',
                                  group, f'interpolated_AU_{pose}')

    for csv in sorted(os.listdir(videos_dir)):

        df = pd.read_csv(os.path.join(videos_dir, csv))
        df = df.drop(columns=['frame'])

        if features[:2] == 'au':
            if 'intensities' in features and 'activations' not in features:
                df = df.iloc[:, :17]  # select the action units intensities
            elif 'intensities' not in features and 'activations' in features:
                df = df.iloc[:, 17:]  # select the action units activations
            elif 'intensities' not in features and 'activations' not in features:
                raise ValueError('Select the type of au features you want to use')

        if features == 'delaunay' and pose != '':
            df = df.drop(columns=['yaw'])

        videos.append(df.mean(axis=0).values)

    labels = get_y(group, pose, emotion)

    return videos, labels


def subsampling(X, y):
    ones = sum(y)
    zeros = len(y) - ones

    X_neg = []
    X_pos = []
    for i, val in enumerate(y):
        if val == 0:
            X_neg.append(X[i])
        else:
            X_pos.append(X[i])

    if ones > zeros:
        samples_needed = zeros

        X_resample = resample(X_pos, replace=False, n_samples=samples_needed)
        new_X = X_resample + X_neg
    else:
        samples_needed = ones

        X_resample = resample(X_neg, replace=False, n_samples=samples_needed)
        new_X = X_pos + X_resample

    new_y = []
    for i in range(samples_needed):
        new_y.append(1)
    for i in range(samples_needed):
        new_y.append(0)

    return np.asarray(new_X, dtype=np.ndarray), np.asarray(new_y)


emotions = ['anger', 'disgust', 'fear',
            'happiness', 'sadness', 'surprise', 'valence']

model, selected_emotion, pose, features = sys.argv[0][:-3], sys.argv[1], sys.argv[2], sys.argv[3]
print(f'Target: {selected_emotion}')
print(f'Pose: {pose}')
print(f'Features: {features}')

emotion_index = emotions.index(selected_emotion)
X, y = read_data('train', pose, emotions[emotion_index], features)
X_val, y_val = read_data('dev', pose, emotions[emotion_index], features)
X_test, y_test = read_data('test', pose, emotions[emotion_index], features)

# Add validation data to train data
X += X_val
y += y_val

X, X_test, y, y_test = np.asarray(X, dtype=np.ndarray), np.asarray(
    X_test, dtype=np.ndarray), np.asarray(y), np.asarray(y_test)

# create the pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC())  # best params: C=10, gamma=62, kernel='sigmoid'
])


# GridSearch per cercare l'intorno su cui effettuare la randomizedSearch
if features == 'delaunay':
    params = {'classifier__C': [i for i in range(10, 31, 1)],
            'classifier__gamma': [i for i in range(60, 81, 1)],
            'classifier__kernel': ['ploy', 'sigmoid', 'rbf']
            }

elif features == 'au_intensities_activations':
    params = {'classifier__C': [i for i in range(1, 10, 1)],
            'classifier__gamma': [i for i in range(0, 20, 1)],
            'classifier__kernel': ['poly']
            }

elif features == 'au_intensities':  # C=471, gamma=260, kernel='sigmoid'
    params = {'classifier__C': [i for i in range(461, 482, 1)],
            'classifier__gamma': [i for i in range(250, 271, 1)],
            'classifier__kernel': ['sigmoid']
            }

elif features == 'au_activations':  # C=1, gamma=1030, kernel='sigmoid'
    params = {'classifier__C': [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200],
            'classifier__gamma': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 1030],
            'classifier__kernel': ['sigmoid']
            }

X, y = subsampling(X, y)
gridsearch = GridSearchCV(pipe, params).fit(X,y)  # fit the model
print(f'Best params: {gridsearch.best_params_}')


# RandomizedSearch
# if features == 'delaunay':
#     params = {'classifier__C': stats.uniform(loc=10, scale=10),
#               'classifier__gamma': stats.uniform(loc=60, scale=10),
#               'classifier__kernel': ['sigmoid']
#               }

# elif features == 'au_intensities_activations':
#     params = {'classifier__C': stats.uniform(loc=0, scale=2),
#               'classifier__gamma': stats.uniform(loc=0, scale=2),
#               'classifier__kernel': ['poly']
#               }

# elif features == 'au_intensities':
#     params = {'classifier__C': stats.uniform(loc=0, scale=2),
#               'classifier__gamma': stats.uniform(loc=0, scale=2),
#               'classifier__kernel': ['poly']
#               }

# X, y = subsampling(X, y)
# randomsearch = RandomizedSearchCV(
#     pipe, params, n_iter=100).fit(X, y)  # fit the model
# print(f'Best params: {randomsearch.best_params_}')

sys.exit()

num_iter = 100
all_pred = []

for i in range(num_iter):

    X, y = subsampling(X, y)

    n_iter = 100 if features[:2] == 'au' else 500
    randomsearch = RandomizedSearchCV(
        pipe, params, n_iter=n_iter).fit(X, y)  # fit the model

    y_pred = randomsearch.predict(X_test)
    all_pred.append(y_pred)

all_pred = np.asarray(all_pred)
final_pred, _ = stats.mode(all_pred)  # voting
final_pred = final_pred[0]

# print(
#     f"Cohen Kappa score is: {cohen_kappa_score(y_test, final_pred, weights='linear')}")
# print("classification report:")
# print(classification_report(y_test, final_pred))

out_dir = os.path.join('results', model, features, selected_emotion, pose)
os.makedirs(out_dir, exist_ok=True)
print(out_dir)

clf_report = classification_report(y_test, final_pred, output_dict=True)
pd.DataFrame(clf_report).T.to_csv(os.path.join(out_dir, 'classification_report.csv'))

sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True)
plt.savefig(os.path.join(out_dir, 'classification_report.png'))

plot_confusion_matrix(estimator=randomsearch, X=X_test,
                      y_true=y_test, normalize='true', cmap='Blues')
plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'))
