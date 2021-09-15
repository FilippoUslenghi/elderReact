import os
import sys
import json
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.metrics import classification_report
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, GRU, Dense, Masking
from tensorflow.keras.preprocessing import sequence
from tensorflow.python.keras.layers.core import Dropout
matplotlib.use('agg')

def load_group(selected_emotion, group):
    
    # load x
    x_dir = os.path.join('dataset_net', 'Features', group, 'delaunay_pose_')

    matrix_list = []
    for csv in sorted(os.listdir(x_dir)):
        df = pd.read_csv(os.path.join(x_dir, csv)).drop(columns='frame') # remove the `frame` column
        matrix_list.append(df.values)
    
    flatten_matrix_list = [matrix.flatten() for matrix in matrix_list] # flatten the matrixes
    # pad the flatten matrix for a length of 748 (biggest n_timesteps in the dataset) by n_features
    padded_flatten_matrix_list = sequence.pad_sequences(flatten_matrix_list, maxlen=748*len(df.columns), padding='post', dtype=np.float64, value=-1)
    padded_matrix_list = [flatten_matrix.reshape(748,-1) for flatten_matrix in padded_flatten_matrix_list] # reshape the flatten matrices to the original shape
    x_group = np.stack(padded_matrix_list, axis=0) # create a 3D matrix of the stacked dataframes with LSTM shape (n_samples, n_timestep, n_features)


    # load y
    y_path = os.path.join('dataset_net', 'Annotations', f'{group}_labels.txt')
    y_df = pd.read_csv(y_path, delim_whitespace=True, header=None).drop(columns=[0, 7])  # drop name and gender columns
    y_df.columns = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'valence']

    y_group = y_df[selected_emotion].values

    if selected_emotion == 'valence':
      # encode valence in a binary value
      y_group = np.asarray([int(y>=4) for y in y_group])

    return x_group, y_group

def load_dataset(selected_emotion):

    dataset = []
    groups = ['train', 'dev', 'test']

    for group in groups:
        x,y = load_group(selected_emotion, group)
        dataset.append(x)
        dataset.append(y)   

    return dataset

def subsampling(X, y):
    ones = sum(y)
    zeros = len(y) - ones

    X_neg = []
    X_pos = []
    for i, val in enumerate(y):
        if val == 0:
            X_neg.append(X[i,:,:])
        else:
            X_pos.append(X[i,:,:])

    if ones > zeros:
        samples_needed = zeros

        X_resample = resample(X_pos, replace=False, n_samples=samples_needed)
        new_X = X_resample + X_neg
        new_X = np.stack(new_X, axis=0)

    else:
        samples_needed = ones

        X_resample = resample(X_neg, replace=False, n_samples=samples_needed)
        new_X = X_pos + X_resample
        new_X = np.stack(new_X, axis=0)

    new_y = []
    for i in range(samples_needed):
        new_y.append(1)
    for i in range(samples_needed):
        new_y.append(0)

    return new_X, np.asarray(new_y)
    
def main(selected_emotion):
    x_train, y_train, x_dev, y_dev, x_test, y_test = load_dataset(selected_emotion)
    x_train = np.vstack([x_train, x_dev])  # add dev data to train data
    y_train = np.append(y_train, y_dev, axis=0)  # add dev labels to train labels
    new_x_train, new_y_train = subsampling(x_train, y_train)

    verbose, epochs, batch_size = 0, 15, 64
    n_timesteps, n_features, n_outputs = new_x_train.shape[1], new_x_train.shape[2], 1


    model = Sequential()
    model.add(Masking(mask_value=-1, input_shape=(n_timesteps, n_features)))
    model.add(LSTM(100, dropout=0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_outputs, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit network
    model.fit(new_x_train, new_y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

    # Evaluate
    y_pred = model.predict(x_test)
    y_pred_class = np.where(y_pred>0.5, 1, 0)

    out_dir = os.path.join('results', 'LSTM', 'delaunay', selected_emotion)
    os.makedirs(out_dir, exist_ok=True)

    data = {'cohen_kappa': cohen_kappa_score(y_test, y_pred_class, weights='linear')}
    with open(os.path.join(out_dir, 'cohen_kappa.json'), 'w') as f:
        json.dump(data, f)

    clf_report = classification_report(
        y_test, y_pred_class, labels=[0, 1], output_dict=True)
    pd.DataFrame(clf_report).T.to_csv(os.path.join(out_dir, 'classification_report.csv'))

    sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True)
    plt.savefig(os.path.join(out_dir, 'classification_report.png'))
    plt.close()

    confusion_mtx = confusion_matrix(y_test, y_pred_class, normalize='true')
    sns.heatmap(confusion_mtx, annot=True, fmt='.2g', cmap='Blues')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'))


if __name__ == '__main__':

    selected_emotion = sys.argv[1]
    main(selected_emotion=selected_emotion)