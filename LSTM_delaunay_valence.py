import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_group(group):
    
    # load x
    x_dir = os.path.join('dataset_net', group, 'delaunay_pose_padded')

    matrix_list = list()
    for csv in os.listdir(x_dir):
        df_values = pd.read_csv(os.path.join(x_dir, csv)).values
        matrix_list.append(df_values)
    
    x_group = np.dstack(matrix_list)[:,1:,:] # remove the `frame` column

    # load y
    y_path = os.path.join('dataset', 'ElderReact-master', 'Annotations', f'{group}_labels.txt')
    y_df = pd.read_csv(y_path, delim_whitespace=True, header=None)

    valence = y_df[8].values
    mean = np.mean(valence)
    std_dv = np.std(valence)

    # normilize the valence to a normal distribution
    y_group = np.asarray([(y-mean)/std_dv for y in valence])

    return x_group, y_group

def load_dataset():

    dataset = list()
    groups = ['train', 'dev', 'test']

    for group in groups:
        x,y = load_group(group)
        dataset.append(x)
        dataset.append(y)   

    return dataset

def plot_history(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()


x_train, y_train, x_dev, y_dev, x_test, y_test = load_dataset()
x_train.reshape(x_train.shape[2], x_train.shape[0], x_train.shape[1]) # reshapeing for the LSTM
epochs = 15
batch_size = 64
n_timesteps, n_features = x_train.shape[1], x_train.shape[2]
n_outputs = 1

# early stopping
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

# Create checkpoint callback that will save the best model observed during training for later use
checkpoint_path = "network_checkpoints/cp.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', save_weights_only=True, verbose=1)

# Build the model
model = Sequential()
model.add(LSTM(100, input_shape=(n_timesteps,n_features)))
model.add(Dense(100, activation='relu'))
model.add(Dense(n_outputs, activation='linear'))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

# Train
history = model.fit(x_train, y_train, validation_data=(x_dev, y_dev), epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[es,cp_callback])

plot_history(history)

# Test
_, mse = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)

print('Test: %.3f' % (mse))