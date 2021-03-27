# %% [markdown]
# #Visualizzazione dati

# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from mpl_toolkits.mplot3d import Axes3D
import re

# %%
# ## Visualizzazione delle features di 40 video presi casualmente

# %%
# Creazione delle variabili comuni

base_dir = 'C:\\Users\\us98\\PycharmProjects\\elderReactProject\\myProcessed\\'
videoList = os.listdir(base_dir)
small_videoList = videoList[::15][:-1]
columns = [col.replace(" ", "") for col in pd.read_csv(base_dir + '50_50_4\\50_50_4.csv').columns]
# %% [markdown]
# Visualizzazione della confidence

# %%
print("Confidence nel tempo per ogni video:")

fig, axes = plt.subplots(8, 5, figsize=(25, 20), sharey=True)
axes = axes.flatten()

for i, videoName in enumerate(small_videoList):
    videoCsv = base_dir + videoName + '\\' + videoName + ".csv"
    df = pd.read_csv(videoCsv)
    df.columns = columns
    
    df[['confidence']].plot(ax=axes[i], legend = False)
    axes[i].set(xlabel='Frame Number', ylabel="Confidence")

plt.yticks([x/10 for x in range(11)])
plt.tight_layout()
plt.show()

# %% [markdown]
# Visualizzazione del gaze angle

# %%
print("Gaze angle nel tempo per ogni video:")

fig, axes = plt.subplots(8, 5, figsize=(25, 20), sharey=True)
axes = axes.flatten()

for i, videoName in enumerate(small_videoList):
    videoCsv = base_dir + videoName + '\\' + videoName + '.csv'
    df = pd.read_csv(videoCsv)
    df.columns = columns

    df[['gaze_angle_x', 'gaze_angle_y']].plot(ax=axes[i], legend = False)
    axes[i].set(ylim=[-1.5, 1.5], xlabel='Frame Number', ylabel='Radians', title=videoName)
    axes[i].legend(['gaze_angle_x', 'gaze_angle_y'], loc='lower left')

plt.tight_layout()
plt.show()

# %% [markdown]
# Visualizzazione delle coordinate 3D del *gaze vector*

# %%
print("Coordinate del gaze vector nel tempo per ogni video:")

fig = plt.figure(figsize=(25, 10))
for i, videoName in enumerate(small_videoList):
    videoCsv = base_dir + videoName + '\\' + videoName + '.csv'
    df = pd.read_csv(videoCsv)
    df.columns = columns


    # Plot delle coordinate spaziali del gaze vector
    ax = fig.add_subplot(4, 10, i+1, projection='3d')

    ax.plot(df.gaze_0_x, df.gaze_0_y, df.gaze_0_z, color='blue')
    ax.plot(df.gaze_1_x, df.gaze_1_y, df.gaze_1_z, color='red')
    ax.set_title(videoName)
    ax.set(xlabel='x', ylabel='y', zlabel='z', xticks=[], yticks=[], zticks=[])
    ax.legend(['Leftmost eye', 'Rightmost eye'], fontsize='xx-small')

plt.tight_layout()
plt.show()

# %% [markdown]
# Visualizzazione della media dei *face landmark* in 2D di ogni frame

# %%
print("Media delle coordinate 2D dei face landmark per ogni frame:")

x_regex_pat = re.compile(r'^x_[0-9]+$')
y_regex_pat = re.compile(r'^y_[0-9]+$')

fig, axes = plt.subplots(8, 5, figsize=(15, 15), sharex=True, sharey=True)
axes = axes.flatten()

for i, videoName in enumerate(small_videoList):
    videoCsv = base_dir + videoName + '\\' + videoName + '.csv'
    df = pd.read_csv(videoCsv)
    df.columns = columns

    x_locs = df.columns[df.columns.str.contains(x_regex_pat)]
    y_locs = df.columns[df.columns.str.contains(y_regex_pat)]


    palette = sns.color_palette()
    avg_face_df = pd.DataFrame({'x_locs': df[x_locs].mean(axis=1), 'y_locs': df[y_locs].mean(axis=1)})
    sns.scatterplot(x='x_locs', y='y_locs', data=avg_face_df, marker='+', ax=axes[i])
    axes[i].set(xlim=[0, 1920], ylim=[1080, 0], title=videoName)

plt.tight_layout()
plt.show()

# %% [markdown]
# Visualizzazione della media dei *face landmark* in 2D di ogni frame

# %%
print("Media delle coordinate 3D dei face landmark per ogni frame:")

X_regex_pat = re.compile(r'^X_[0-9]+$')
Y_regex_pat = re.compile(r'^Y_[0-9]+$')
Z_regex_pat = re.compile(r'^Z_[0-9]+$')

fig = plt.figure(figsize=(25, 10))
for i, videoName in enumerate(small_videoList):
    videoCsv = base_dir + videoName + '\\' + videoName + '.csv'
    df = pd.read_csv(videoCsv)
    df.columns = columns

    X_locs = df.columns[df.columns.str.contains(X_regex_pat)]
    Y_locs = df.columns[df.columns.str.contains(Y_regex_pat)]
    Z_locs = df.columns[df.columns.str.contains(Z_regex_pat)]

    df_locs = pd.DataFrame({'X_locs': df[X_locs].mean(axis=1), 'Y_locs': df[Y_locs].mean(axis=1), 'Z_locs': df[Z_locs].mean(axis=1)})

    # Plot delle coordinate spaziali del gaze vector
    ax = fig.add_subplot(4, 10, i+1, projection='3d')

    ax.scatter(df_locs.X_locs, df_locs.Y_locs, df_locs.Z_locs, marker='+')
    ax.set_title(videoName)
    ax.set(xlabel='x', ylabel='y', zlabel='z', xticks=[], yticks=[], zticks=[])

plt.tight_layout()
plt.show()

# %% [markdown]
# Visualizzazione delle coordinate della *head pose location* di ogni frame

# %%
print("Coordinate della head pose location di ogni frame:")

fig = plt.figure(figsize=(25, 10))
for i, videoName in enumerate(small_videoList):
    videoCsv = base_dir + videoName + '\\' + videoName + '.csv'
    df = pd.read_csv(videoCsv)
    df.columns = columns

    # Plot delle coordinate spaziali del gaze vector
    ax = fig.add_subplot(4, 10, i+1, projection='3d')

    ax.scatter(df.pose_Tx, df.pose_Ty, df.pose_Tz, marker='+')
    ax.set_title(videoName)
    ax.set(xlabel='x', ylabel='y', zlabel='z', xticks=[], yticks=[], zticks=[])

# %% [markdown]
# Visualizzazione della * head pose rotation* nel tempo per ogni video

# %%
print("Head pose rotation nel tempo per ogni video:")

fig, axes = plt.subplots(8, 5, figsize=(25, 20), sharey=True)
axes = axes.flatten()

for i, videoName in enumerate(small_videoList):
    videoCsv = base_dir + videoName + '\\' + videoName + '.csv'
    df = pd.read_csv(videoCsv)
    df.columns = columns

    df[['pose_Rx', 'pose_Ry', 'pose_Rz']].plot(ax=axes[i], legend = False)
    axes[i].set(ylim=[-1.5, 1.5], xlabel='Frame Number', ylabel='Radians', title=videoName)
    axes[i].legend(['pose_Rx', 'pose_Ry', 'pose_Rz'], loc='lower left')

plt.tight_layout()
plt.show()

# %%
# Visualizzazione dell'area contenuta dal contorno del volto nel tempo per ogni video

# %%
def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

print("Visualizzazione dell'area del volto per frame:")


