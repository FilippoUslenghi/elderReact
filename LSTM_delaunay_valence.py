import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.keras.preprocessing import sequence

def load_group(group):
    
    # load x
    x_dir = os.path.join('dataset_net', 'Features', group, 'delaunay_pose')

    matrix_list = []
    for csv in sorted(os.listdir(x_dir)):
        df = pd.read_csv(os.path.join(x_dir, csv)).drop(columns='frame') # remove the `frame` column
        matrix_list.append(df.values)
    
    flatten_matrix_list = [matrix.flatten() for matrix in matrix_list] # flatten the matrixes
    padded_flatten_matrix_list = sequence.pad_sequences(flatten_matrix_list, padding='post') # pad the matrix
    flatten_matrix_list = [flatten_matrix.reshape(113,-1).T for flatten_matrix in padded_flatten_matrix_list] # reshape the flatten matrices to the original shape
    x_group = np.dstack(flatten_matrix_list) # create a 3D matrix of the stacked dataframes
    x_group = x_group.reshape(x_group.shape[2], x_group.shape[0], x_group.shape[1]) # reshaping for the LSTM

    # load y
    y_path = os.path.join('dataset_net', 'Annotations', f'{group}_labels.txt')
    y_df = pd.read_csv(y_path, delim_whitespace=True, header=None)
    valence = y_df[8].values

    # normalize the valence to a normal distribution
    mean = np.mean(valence)
    std_dv = np.std(valence)
    y_group = np.asarray([(y-mean)/std_dv for y in valence])

    return x_group, y_group

def load_dataset():

    dataset = []
    groups = ['train', 'dev', 'test']

    for group in groups:
        x,y = load_group(group)
        dataset.append(x)
        dataset.append(y)   

    return dataset

def plot_history(history):
    # Plot training & validation MSE values
    plt.plot(history.history['mean_squared_error'])
    plt.plot(history.history['val_mean_squared_error'])
    plt.title('Model MSE')
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()


x_train, y_train, x_dev, y_dev, x_test, y_test = load_dataset()
epochs = 15
batch_size = 64
n_timesteps, n_features = x_train.shape[1], x_train.shape[2]
n_outputs = 1

# Build the model
model = Sequential()
model.add(LSTM(32, input_shape=(1,n_features)))
model.add(Dropout(0.5))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(n_outputs, activation='linear'))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

# Early stopping
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
# Create checkpoint callback that will save the best model observed during training for later use
checkpoint_path = os.path.join('network_checkpoints','cp.ckpt')
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', save_weights_only=True, verbose=1)

# Train
history = model.fit(x_train, y_train, validation_data=(x_dev, y_dev), epochs=epochs, batch_size=batch_size, verbose=1) #callbacks=[es,cp_callback])

plot_history(history)

# Test
_, mse = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)

print('Test: %.3f' % (mse))