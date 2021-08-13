import pandas as pd
import numpy as np
import os
import sys


model = sys.argv[1]
if model != 'LSTM':

    features_list = ['delaunay', 'au_intensities',
                     'au_activations', 'au_intensities_activations']
    emotions = ['anger', 'disgust', 'fear',
                'happiness', 'sadness', 'surprise', 'valence']
    poses = ['tilted', 'frontal', 'none']
    data = np.ndarray(shape=(14, 12), dtype=np.float32)

    for f, features in enumerate(features_list):
        for e, emotion in enumerate(emotions):
            for p, pose in enumerate(poses):

                base_dir = os.path.join(model, features, emotion, pose)
                df = pd.read_csv(os.path.join(
                    base_dir, 'classification_report.csv'))
                data[e][f*len(poses)+p] = df.at[3, 'f1-score']  # macro avg
                data[e+len(emotions)][f*len(poses)+p] = df.at[4,'f1-score']  # weigthed avg

    data = data.round(2)

    columns = [
        ['delaunay', 'delaunay', 'delaunay', 'au_intensities', 'au_intensities', 'au_intensities', 'au_activations', 'au_activations',
            'au_activations', 'au_intensities_activations', 'au_intensities_activations', 'au_intensities_activations'],
        ['tilted', 'frontal', 'none', 'tilted', 'frontal', 'none',
            'tilted', 'frontal', 'none', 'tilted', 'frontal', 'none']
    ]

    rows = [
        ['macro_avg', 'macro_avg', 'macro_avg', 'macro_avg', 'macro_avg', 'macro_avg', 'macro_avg', 'weighted_avg',
            'weighted_avg', 'weighted_avg', 'weighted_avg', 'weighted_avg', 'weighted_avg', 'weighted_avg'],
        ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'valence',
            'anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'valence']
    ]

    columns_index = pd.MultiIndex.from_arrays(
        columns, names=['features', 'pose'])
    rows_index = pd.MultiIndex.from_arrays(
        rows, names=['f1_score', 'emotion'])
    df = pd.DataFrame(data=data, index=rows_index, columns=columns_index)
    df.to_csv(os.path.join(f'{model}.csv'))

else:

    features_list = ['au_intensities', 'au_activations', 'au_intensities_activations']
    emotions = ['anger', 'disgust', 'fear',
                'happiness', 'sadness', 'surprise', 'valence']
    poses = ['tilted', 'frontal', 'none']
    data = np.ndarray(shape=(14, 3), dtype=np.float32)

    for f, features in enumerate(features_list):
        for e, emotion in enumerate(emotions):

                base_dir = os.path.join(model, features, emotion)
                df = pd.read_csv(os.path.join(
                    base_dir, 'classification_report.csv'))
                data[e][f] = df.at[3, 'f1-score']  # macro avg
                data[e+len(emotions)][f] = df.at[4,'f1-score']  # weigthed avg

    data = data.round(2)

    columns = [
        ['au_intensities', 'au_activations', 'au_intensities_activations']
    ]

    rows = [
        ['macro_avg', 'macro_avg', 'macro_avg', 'macro_avg', 'macro_avg', 'macro_avg', 'macro_avg', 'weighted_avg',
            'weighted_avg', 'weighted_avg', 'weighted_avg', 'weighted_avg', 'weighted_avg', 'weighted_avg'],
        ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'valence',
            'anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'valence']
    ]

    rows_index = pd.MultiIndex.from_arrays(
        rows, names=['f1_score', 'emotion'])
    df = pd.DataFrame(data=data, index=rows_index, columns=columns)
    df.to_csv(os.path.join(f'{model}.csv'))
