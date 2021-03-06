import numpy as np
from sklearn.metrics import cohen_kappa_score
from sklearn.svm import SVC
from sklearn import svm
import os
from xgboost import XGBClassifier
from sklearn.utils import resample
import scipy.io
from sklearn.svm import SVR
from scipy import stats
import json
from sklearn.metrics import roc_auc_score
import math
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from scipy.stats import uniform as sp_rand
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import random
from sklearn.metrics import precision_recall_fscore_support

affects = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprise']
affect_state = 7  # choose the affect to recognize, range 0~5, 7 is valence
if affect_state != 7:
    print(affects[affect_state] + "...")
else:
    print("valence...")
feature_mode = "video"  # audio, video or bimodal
train_path = ''  # change to your training data path
val_path = ''
test_path = ''


def read_data(audio_feat_file, video_feat_file, label_file, affect_state, feature_mode):
    X = []
    y = []

    for i in range(len(audio_feat_file)):
        audio_feat = audio_feat_file[i].strip().split("\t")[1]
        audio_feat = audio_feat.split()
        audio_feat = [float(i) for i in audio_feat]
        audio_feat = audio_feat[:36] + audio_feat[74:110]

        video_feat = video_feat_file[i].strip().split("\t")[1]
        video_feat = video_feat.split()
        v = [float(i) for i in video_feat]
        video_feat = v[0:8] + v[288:294] + v[634:717] + v[997:1003] + v[1343:]

        if feature_mode == "audio":
            feat = audio_feat
        elif feature_mode == "video":
            feat = video_feat
        elif feature_mode == "bimodal":
            feat = audio_feat + video_feat

        label = label_file[i].split(" ")[1:]

        if affect_state == 7:
            label = float(label[affect_state][:-2])
        else:
            label = int(label[affect_state])

        X.append(feat)
        y.append(label)

    return X, y


base_dir = os.path.join('')

f = open(os.path.join(base_dir, "Features/Audio_feat/train_audio.txt")).readlines()
g = open(os.path.join(base_dir, "Features/Video_feat/train_video.txt")).readlines()
h = open(os.path.join(base_dir, "Annotations/train_labels.txt")).readlines()
X, y = read_data(f, g, h, affect_state, feature_mode)

f = open(os.path.join(base_dir, "Features/Audio_feat/dev_audio.txt")).readlines()
g = open(os.path.join(base_dir, "Features/Video_feat/dev_video.txt")).readlines()
h = open(os.path.join(base_dir, "Annotations/dev_labels.txt")).readlines()
X_val, y_val = read_data(f, g, h, affect_state, feature_mode)

f = open(os.path.join(base_dir, "Features/Audio_feat/test_audio.txt")).readlines()
g = open(os.path.join(base_dir, "Features/Video_feat/test_video.txt")).readlines()
h = open(os.path.join(base_dir, "Annotations/test_labels.txt")).readlines()
X_test, y_test = read_data(f, g, h, affect_state, feature_mode)

X = np.asarray(X)
X_val = np.asarray(X_val)
X_test = np.asarray(X_test)
# print(np.where(np.isnan(X)))
# print(np.where(np.isnan(X_val)))
# print(np.where(np.isnan(X_test)))
X_test = np.delete(X_test, 289, axis=0)  # delete NaN value
X_test = np.delete(X_test, 43, axis=0)
y_test.pop(289)
y_test.pop(43)

# feature normalization
train_min = X.min(axis=0)
train_max = X.max(axis=0)

diff = train_min - train_max
X = X - train_min
X = X / diff  # .T

X_val = X_val - train_min
X_val = X_val / diff

X_test = X_test - train_min
X_test = X_test / diff

if affect_state == 7:
    y = [int(value >= 4) for value in y]
    y_val = [int(value >= 4) for value in y_val]
    y_test = [int(value >= 4) for value in y_test]

def subsampling(X, y):
    samples_needed = 0
    for num in y:
        if num == 1:
            samples_needed += 1

    X_neg = []
    X_pos = []
    for i, val in enumerate(y):
        if val == 0:
            X_neg.append(X[i])
        else:
            X_pos.append(X[i])

    X_resample = resample(X_neg, replace=True, n_samples=samples_needed)

    new_X = X_pos + X_resample

    new_y = []
    for i in range(samples_needed):
        new_y.append(1)
    for i in range(samples_needed):
        new_y.append(0)

    return new_X, new_y


num_iter = 100
all_pred = []
all_prob = []
all_f1 = 0
clf_mode = "xgboost"  # svm, gnb, xgboost, dummy
mode = "test"  # val or test. val mode is for searching for hyperparameters
for i in range(num_iter):

    new_X, new_y = subsampling(X, y)
    new_X = np.asarray(new_X)
    new_y = np.asarray(new_y)

    if clf_mode == "gnb":
        clf = GaussianNB()
        clf.fit(new_X, new_y)

    elif clf_mode == "svm":
        e = -4  # hyperparameters to be tuned
        c = math.pow(10, e)
        clf = SVC(gamma='scale', C=c)
        clf.fit(new_X, new_y)

    elif clf_mode == "xgboost":
        clf = XGBClassifier()
        clf.fit(new_X, new_y, eval_metric='logloss')

    elif clf_mode == "dummy":
        clf = DummyClassifier()
        clf.fit(new_X, new_y)

    # clf.fit(new_X, new_y)

    if mode == "val":
        y_pred = clf.predict(X_val)  # X_val or X_test

    elif mode == "test":
        y_pred = clf.predict(X_test)

    all_pred.append(y_pred)

all_pred = np.asarray(all_pred)
final_pred, _ = stats.mode(all_pred)  # voting
final_pred = final_pred[0]
# print(final_pred)

# print(f"accuracy score is: {accuracy_score(y_test, final_pred)}")
print(
    f"Cohen Kappa score is: {cohen_kappa_score(y_test, final_pred, weights='linear')}")
print("classification report:")
print(classification_report(y_test, final_pred))

# count = 0
# for i in range(len(y_test)):
# 	if y_test[i] == 1:
# 		count+=1

# print("positive cases..")
# print(count)
# print("negative cases..")
# print(len(y_test)-count)
