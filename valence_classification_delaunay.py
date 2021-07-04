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
    # pad the flatten matrix for a length of 748 (biggest n_timesteps in the dataset) by 113 (n_features)\1
    padded_flatten_matrix_list = sequence.pad_sequences(flatten_matrix_list, maxlen=748*113, padding='post', dtype=np.float64)
    padded_matrix_list = [flatten_matrix.reshape(748,-1) for flatten_matrix in padded_flatten_matrix_list] # reshape the flatten matrices to the original shape
    x_group = np.stack(padded_matrix_list, axis=0) # create a 3D matrix of the stacked dataframes with LSTM shape (n_samples, n_timestep, n_features)

    # load y
    y_path = os.path.join('dataset_net', 'Annotations', f'{group}_labels.txt')
    y_df = pd.read_csv(y_path, delim_whitespace=True, header=None)
    valence = y_df[8].values

    # encode valence in a binary value
    y_group = np.asarray([int(y>=3.5) for y in valence])

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
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()


x_train, y_train, x_dev, y_dev, x_test, y_test = load_dataset()
print(x_train[0,:,0])
import sys; sys.exit()
epochs = 15
batch_size = 8
n_timesteps, n_features = x_train.shape[1], x_train.shape[2]
n_outputs = 1

# Build the model
model = Sequential()
model.add(LSTM(32, input_shape=(n_timesteps, n_features)))
model.add(Dropout(0.5))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(n_outputs, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Early stopping
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
# Create checkpoint callback that will save the best model observed during training for later use
checkpoint_path = os.path.join('network_checkpoints','cp.ckpt')
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', save_weights_only=True, verbose=1)

# Train
history = model.fit(x_train, y_train, validation_data=(x_dev, y_dev), epochs=epochs, batch_size=batch_size, verbose=1) #callbacks=[es,cp_callback])

plot_history(history)

# Test
_, accuracy = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)

print('Test: %.3f accuracy' % (accuracy))