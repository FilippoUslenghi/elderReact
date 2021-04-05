# %% [markdown]
# # Creazione di un subset di video

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
base_dir = 'C:\\Users\\us98\\PycharmProjects\\elderReactProject\\myProcessed\\'
all_videos = os.listdir(base_dir)
columns = [col.replace(" ", "") for col in pd.read_csv(base_dir + '50_50_4\\50_50_4.csv').columns]

# %% [markdown]
# # Confidence
# Per ogni video voglio vedere che valore assume la varianza della confidence.
# Sono poi interessato ad imporre una threshold su di essa in base alla quale selezionare i video che manterrò.

# %%
print("Varianza della confidence per ogni video:")

plt.figure(figsize=(10, 8))

all_videos_var_list = []
for videoName in all_videos:
    videoCsv = base_dir + videoName + '\\' + videoName + ".csv"
    df = pd.read_csv(videoCsv)
    df.columns = columns
    
    all_videos_var_list.append(df.confidence.var())

all_videos_var_df = pd.DataFrame({'video': all_videos, 'variance': all_videos_var_list})

plt.barh(all_videos_var_df.index, all_videos_var_df.variance)
plt.title('Variance trend in the dataset')
plt.xlabel='Videos'
plt.ylabel='Variance'
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Creazione del subset di video
# #### Inizializzo una threshold

# %%
threshold = all_videos_var_df.mean()[0]
print('Threshold: ', threshold)

# %% [markdown]
# #### ...e la applico
# %%
print('Creazione...')

good_videos = []
good_videos_var_list = []
for i, videoName in enumerate(all_videos):
    videoCsv = base_dir + videoName + '\\' + videoName + ".csv"
    df = pd.read_csv(videoCsv)
    df.columns = columns

    if df.confidence.var()>=threshold: continue

    good_videos.append(videoName)
    good_videos_var_list.append(df.confidence.var())

good_videos_var_df = pd.DataFrame({'video': good_videos, 'variance': good_videos_var_list})

print('Numero di video:', len(good_videos_var_df))
good_videos_var_df.head()

# %% [markdown]
# #### Osserviamo l'andamento della confidence di un sottoinsieme dei video con varianza superiore alla media

# %%
videos_above_avg = good_videos_var_df[good_videos_var_df.variance>=good_videos_var_df.variance.mean()]
n_videos_above_avg = len(videos_above_avg)
print('Numero di video con varianza sopra la media:', n_videos_above_avg)

fig, axes = plt.subplots(8, 5, figsize=(25, 20), sharey=True)
axes = axes.flatten()

for i, videoName in enumerate(videos_above_avg.video[::4]):
    videoCsv = base_dir + videoName + '\\' + videoName + ".csv"
    df = pd.read_csv(videoCsv)
    df.columns = columns

    df[['confidence']].plot(ax=axes[i], legend = False)
    axes[i].set(xlabel='Frame Number', ylabel="Confidence", title=videoName)

    if i==39: break

plt.ylim(0,1)
plt.yticks([x/10 for x in range(11)])
plt.tight_layout()
plt.show()

# %% [markdown]
# #### Ora nel caso di varianza inferiore alla media

# %%
videos_below_avg = good_videos_var_df[good_videos_var_df.variance<good_videos_var_df.variance.mean()]
n_videos_below_avg = len(videos_below_avg)
print('Numero di video con varianza sotto la media:', n_videos_below_avg)

fig, axes = plt.subplots(8, 5, figsize=(25, 20), sharey=True)
axes = axes.flatten()

for i,videoName in enumerate(videos_below_avg.video[::4]):
    videoCsv = base_dir + videoName + '\\' + videoName + ".csv"
    df = pd.read_csv(videoCsv)
    df.columns = columns
    
    df[['confidence']].plot(ax=axes[i], legend = False)
    axes[i].set(xlabel='Frame Number', ylabel="Confidence", title=videoName)

    if i==39: break

plt.ylim(0,1)
plt.yticks([x/10 for x in range(11)])
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Visualizzazione feature del subset
# ### Gaze angle
# #### Video con varianza sopra la media

# %%
print("Gaze angle nel tempo di alcuni video sopra la media:")

fig, axes = plt.subplots(8, 5, figsize=(25, 20), sharey=True)
axes = axes.flatten()

for i, videoName in enumerate(videos_above_avg.video[::4]):
    videoCsv = base_dir + videoName + '\\' + videoName + '.csv'
    df = pd.read_csv(videoCsv)
    df.columns = columns

    df[['gaze_angle_x', 'gaze_angle_y']].plot(ax=axes[i], legend = False)
    axes[i].set(ylim=[-1.5, 1.5], xlabel='Frame Number', ylabel='Radians', title=videoName)
    axes[i].legend(['gaze_angle_x', 'gaze_angle_y'], loc='lower left')

    if i==39: break

plt.tight_layout()
plt.show()

# %% [markdown]
# #### Video con varianza sotto la media

# %%
print("Gaze angle nel tempo di alcuni video sotto la media:")

fig, axes = plt.subplots(8, 5, figsize=(25, 20), sharey=True)
axes = axes.flatten()

for i, videoName in enumerate(videos_below_avg.video[::4]):
    videoCsv = base_dir + videoName + '\\' + videoName + '.csv'
    df = pd.read_csv(videoCsv)
    df.columns = columns

    df[['gaze_angle_x', 'gaze_angle_y']].plot(ax=axes[i], legend = False)
    axes[i].set(ylim=[-1.5, 1.5], xlabel='Frame Number', ylabel='Radians', title=videoName)
    axes[i].legend(['gaze_angle_x', 'gaze_angle_y'], loc='lower left')

    if i==39: break

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Gaze angle
# #### Video con varianza sopra la media

# %%
print("Coordinate 3D del gaze vector nel tempo di alcuni video sopra la media:")

fig = plt.figure(figsize=(25,30))
for i, videoName in enumerate(videos_above_avg.video[::4]):
    videoCsv = base_dir + videoName + '\\' + videoName + '.csv'
    df = pd.read_csv(videoCsv)
    df.columns = columns

    # Plot delle coordinate spaziali del gaze vector
    ax = fig.add_subplot(8, 5, i+1, projection='3d')

    ax.plot(df.gaze_0_x, df.gaze_0_y, df.gaze_0_z, color='blue')
    ax.plot(df.gaze_1_x, df.gaze_1_y, df.gaze_1_z, color='red')
    ax.set_title(videoName, fontdict={'fontsize':16})
    ax.set(xlabel='x', ylabel='y', zlabel='z', xticks=[], yticks=[], zticks=[])
    ax.legend(['Leftmost eye', 'Rightmost eye'], fontsize='xx-small')

    if i==39: break

plt.tight_layout()
plt.show()

# %% [markdown]
# #### Video con varianza sotto la media

# %%
print("Coordinate 3D del gaze vector nel tempo di alcuni video sotto la media:")

fig = plt.figure(figsize=(25, 30))
for i, videoName in enumerate(videos_below_avg.video[::4]):
    videoCsv = base_dir + videoName + '\\' + videoName + '.csv'
    df = pd.read_csv(videoCsv)
    df.columns = columns

    # Plot delle coordinate spaziali del gaze vector
    ax = fig.add_subplot(8, 5, i+1, projection='3d')

    ax.plot(df.gaze_0_x, df.gaze_0_y, df.gaze_0_z, color='blue')
    ax.plot(df.gaze_1_x, df.gaze_1_y, df.gaze_1_z, color='red')
    ax.set_title(videoName, fontdict={'fontsize':16})
    ax.set(xlabel='x', ylabel='y', zlabel='z', xticks=[], yticks=[], zticks=[])
    ax.legend(['Leftmost eye', 'Rightmost eye'], fontsize='xx-small')
    
    if i==39: break

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Head pose rotation
# #### Video con varianza sopra la media

# %%
print("Head pose rotation nel tempo di alcuni video sopra la media:")

fig, axes = plt.subplots(8, 5, figsize=(25, 20), sharey=True)
axes = axes.flatten()

for i, videoName in enumerate(videos_above_avg.video[::4]):
    videoCsv = base_dir + videoName + '\\' + videoName + '.csv'
    df = pd.read_csv(videoCsv)
    df.columns = columns

    df[['pose_Rx', 'pose_Ry', 'pose_Rz']].plot(ax=axes[i], legend = False)
    axes[i].set(ylim=[-1.5, 1.5], xlabel='Frame Number', ylabel='Radians', title=videoName)
    axes[i].legend(['pose_Rx', 'pose_Ry', 'pose_Rz'], loc='lower left')

    if i==39: break

plt.tight_layout()
plt.show()

# %% [markdown]
# #### Video con varianza sotto la media

# %%
print("Head pose rotation nel tempo di alcuni video sotto la media:")

fig, axes = plt.subplots(8, 5, figsize=(25, 20), sharey=True)
axes = axes.flatten()

for i, videoName in enumerate(videos_below_avg.video[::4]):
    videoCsv = base_dir + videoName + '\\' + videoName + '.csv'
    df = pd.read_csv(videoCsv)
    df.columns = columns

    df[['pose_Rx', 'pose_Ry', 'pose_Rz']].plot(ax=axes[i], legend = False)
    axes[i].set(ylim=[-1.5, 1.5], xlabel='Frame Number', ylabel='Radians', title=videoName)
    axes[i].legend(['pose_Rx', 'pose_Ry', 'pose_Rz'], loc='lower left')

    if i==39: break

plt.tight_layout()
plt.show()

# %%
