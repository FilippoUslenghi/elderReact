import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Masking
from tensorflow.keras.preprocessing import sequence
from tensorflow.python.keras.layers.core import Dropout
matplotlib.use('agg')


def load_group(selected_emotion, group, intensities, activations):

    # load x
    x_dir = os.path.join('dataset_net', 'Features', group, 'interpolated_AU_')

    matrix_list = []
    for csv in sorted(os.listdir(x_dir)):
        df = pd.read_csv(os.path.join(x_dir, csv)).drop(
            columns='frame')  # remove the `frame` column

        if intensities and not activations:
            df = df.iloc[:, :17]  # select the action units intensities
        elif not intensities and activations:
            df = df.iloc[:, 17:]  # select the action units activations
        elif not intensities and not activations:
            raise ValueError(
                'Intensities and activations cannot be both False')

        matrix_list.append(df.values)

    flatten_matrix_list = [matrix.flatten()
                           for matrix in matrix_list]  # flatten the matrixes
    # pad the flatten matrix for a length of 748 (biggest n_timesteps in the dataset) by 113 (n_features)\1
    padded_flatten_matrix_list = sequence.pad_sequences(
        flatten_matrix_list, maxlen=748*len(df.columns), padding='post', dtype=np.float64, value=-1)
    # reshape the flatten matrices to the original shape
    padded_matrix_list = [flatten_matrix.reshape(
        748, -1) for flatten_matrix in padded_flatten_matrix_list]
    # create a 3D matrix of the stacked dataframes with LSTM shape (n_samples, n_timestep, n_features)
    x_group = np.stack(padded_matrix_list, axis=0)

    # load y
    y_path = os.path.join('dataset_net', 'Annotations', f'{group}_labels.txt')
    y_df = pd.read_csv(y_path, delim_whitespace=True, header=None).drop(
        columns=[0, 7])  # drop name and gender columns
    y_df.columns = ['anger', 'disgust', 'fear',
                    'happiness', 'sadness', 'surprise', 'valence']

    y_group = y_df[selected_emotion].values

    if selected_emotion == 'valence':
        # encode valence in a binary value
        y_group = np.asarray([int(y >= 4) for y in y_group])

    return x_group, y_group


def load_dataset(selected_emotion, intensities, activations):

    dataset = []
    groups = ['train', 'dev', 'test']

    for group in groups:
        x, y = load_group(selected_emotion, group, intensities, activations)
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
            X_neg.append(X[i, :, :])
        else:
            X_pos.append(X[i, :, :])

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


def main(selected_emotion, intensities, activations):

    x_train, y_train, x_dev, y_dev, x_test, y_test = load_dataset(
        selected_emotion, intensities, activations)
    new_x_train, new_y_train = subsampling(x_train, y_train)

    if intensities and activations:
        features = 'intensities_activations'

    elif intensities and not activations:
        features = 'intensities'

    elif not intensities and activations:
        features = 'activations'

    epochs = 15
    batch_size = 8
    n_timesteps, n_features = x_train.shape[1], x_train.shape[2]
    n_outputs = 1

    # Build the model
    model = Sequential()
    model.add(Masking(mask_value=-1, input_shape=(n_timesteps, n_features)))
    model.add(Bidirectional(LSTM(64, input_shape=(n_timesteps, n_features), return_sequences=True, dropout=0.5,
                                 kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), activity_regularizer=regularizers.l2(1e-5))))
    model.add(Bidirectional(LSTM(64, kernel_regularizer=regularizers.l1_l2(
        l1=1e-5, l2=1e-4), activity_regularizer=regularizers.l2(1e-5), dropout=0.5)))
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l1_l2(
        l1=1e-5, l2=1e-4), activity_regularizer=regularizers.l2(1e-5)))
    model.add(Dropout(0.7))
    model.add(Dense(n_outputs, activation='sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    # Early stopping
    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', mode='max', verbose=1, patience=5)
    # Create checkpoint callback that will save the best model observed during training for later use
    checkpoint_path = os.path.join(
        'network_checkpoints', 'classification_cp.ckpt')
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, monitor='val_loss', save_weights_only=True, verbose=0)

    # Train
    history = model.fit(new_x_train, new_y_train, validation_data=(
        x_dev, y_dev), epochs=epochs, batch_size=batch_size, verbose=0)  # , callbacks=[es,cp_callback])

    # Evaluate
    y_pred = model.predict(x_test)
    y_pred_class = np.where(y_pred > 0.5, 1, 0)
    # print(classification_report(y_test, y_pred_class, labels=[0, 1]))

    out_dir = os.path.join('results', 'LSTM', features, selected_emotion)
    os.makedirs(out_dir, exist_ok=True)

    clf_report = classification_report(
        y_test, y_pred_class, labels=[0, 1], output_dict=True)
    sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True)
    plt.savefig(os.path.join(out_dir, 'classification_report.png'))
    plt.close()

    confusion_mtx = confusion_matrix(y_test, y_pred_class, normalize='true')
    sns.heatmap(confusion_mtx, annot=True, fmt='.2g', cmap='Blues')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'))


if __name__ == '__main__':

    selected_emotion, features = sys.argv[1], sys.argv[2]

    intensities = 1 if 'intensities' in features else 0
    activations = 1 if 'activations' in features else 0

    main(selected_emotion=selected_emotion,
         intensities=intensities, activations=activations)
