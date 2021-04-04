# %% [markdown]
# # Visualizzazione dati

# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from mpl_toolkits.mplot3d import Axes3D
import re

# %% [markdown]
# ## Visualizzazione delle features di 40 video presi casualmente

# %%
# Creazione delle variabili comuni
base_dir = 'C:\\Users\\us98\\PycharmProjects\\elderReactProject\\myProcessed\\'
videoList = os.listdir(base_dir)
small_videoList = videoList[::15][:-1]
columns = [col.replace(" ", "") for col in pd.read_csv(base_dir + '50_50_4\\50_50_4.csv').columns]

# %% [markdown]
# #### Visualizzazione della confidence

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
# #### Visualizzazione del gaze angle

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
# #### Visualizzazione delle coordinate 3D del *gaze vector*

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
# #### Visualizzazione della media dei *face landmark* in 2D di ogni frame

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
# #### Visualizzazione della media dei *face landmark* in 3D di ogni frame

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
# #### Visualizzazione delle coordinate 3D della *head pose location* di ogni frame

# %%
print("Coordinate 3D della head pose location di ogni frame:")

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
# #### Visualizzazione della *head pose rotation* nel tempo per ogni video

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

# %% [markdown]
# #### Visualizzazione dell'area contenuta dal contorno del volto, nel tempo, per ogni video

# %%
def polyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

# %%
print("Visualizzazione dell'area del volto per frame:")

# Per costruire l'area del volto le coordinate devono seguire questo pattern -> [x_0,...,x_16,x_26,...,x_17]
# ----------------------------------------------------------------------------- [y_0,...,y_16,y_26,...,y_17]
face_coordinates_pattern_x = ['x_' + str(x) for x in range(17)] + ['x_' + str(x) for x in range(17, 27)][::-1]
face_coordinates_pattern_y = ['y_' + str(y) for y in range(17)] + ['y_' + str(y) for y in range(17, 27)][::-1]

fig, axes = plt.subplots(8, 5, figsize=(25, 15))
axes = axes.flatten()

for i, videoName in enumerate(small_videoList):
    videoCsv = base_dir + videoName + '\\' + videoName + '.csv'
    df = pd.read_csv(videoCsv)
    df.columns = columns

    # Face
    face_x_points_df = df[face_coordinates_pattern_x]
    face_y_points_df = df[face_coordinates_pattern_y]
    face_area_for_all_frames = np.zeros(shape=len(df))

    for j in range(len(df)):

        # Face
        face_x_points = face_x_points_df.iloc[j].to_numpy()
        face_y_points = face_y_points_df.iloc[j].to_numpy()
        face_area_frame_j = polyArea(face_x_points, face_y_points)
        face_area_for_all_frames[j] = face_area_frame_j

    face_area_df = pd.DataFrame({'frame': df.frame, 'face_area': face_area_for_all_frames})
    palette = sns.color_palette()
    sns.lineplot(x='frame', y='face_area', data=face_area_df, ax=axes[i])
    axes[i].set(xlabel='Frame', ylabel='Area', title=videoName)

plt.tight_layout()
plt.show()

# %% [markdown]
# #### Visualizzazione dell'area contenuta dal contorno della bocca, nel tempo, per ogni video

# %%
print("Visualizzazione dell'area della bocca per frame:")

# Per costruire l'area della bocca le coordinate devono seguire questo pattern -> [x_48,...,x_59]
# ------------------------------------------------------------------------------- [y_46,...,y_59]
mouth_coordinates_pattern_x = ['x_' + str(x) for x in range(48,60)]
mouth_coordinates_pattern_y = ['y_' + str(y) for y in range(48,60)]

fig, axes = plt.subplots(8, 5, figsize=(25, 15))
axes = axes.flatten()

for i, videoName in enumerate(small_videoList):
    videoCsv = base_dir + videoName + '\\' + videoName + '.csv'
    df = pd.read_csv(videoCsv)
    df.columns = columns

    # Mouth
    mouth_x_points_df = df[mouth_coordinates_pattern_x]
    mouth_y_points_df = df[mouth_coordinates_pattern_y]
    mouth_area_for_all_frames = np.zeros(shape=len(df))
    
    for j in range(len(df)):

        # Mouth
        mouth_x_points = mouth_x_points_df.iloc[j].to_numpy()
        mouth_y_points = mouth_y_points_df.iloc[j].to_numpy()
        mouth_area_frame_j = polyArea(mouth_x_points, mouth_y_points)
        mouth_area_for_all_frames[j] = mouth_area_frame_j

    mouth_area_df = pd.DataFrame({'frame': df.frame, 'mouth_area': mouth_area_for_all_frames})
    palette = sns.color_palette()
    sns.lineplot(x='frame', y='mouth_area', data=mouth_area_df, ax=axes[i])
    axes[i].set(xlabel='Frame', ylabel='Area', title=videoName)

plt.tight_layout()
plt.show()

# %% [markdown]
# #### Visualizzazione dell'area contenuta dal contorno degli occhi, nel tempo, per ogni video

# %%
print("Visualizzazione dell'area degli occhi per frame:")

# Per costruire l'area degli occhi le coordinate devono seguire questo pattern -> [eye_lmk_x_8,...,eye_lmk_x_19]
# ------------------------------------------------------------------------------- [eye_lmk_y_8,...,eye_lmk_y_19]
eye1_coordinates_pattern_x = ['eye_lmk_x_' + str(x) for x in range(8,20)]
eye1_coordinates_pattern_y = ['eye_lmk_y_' + str(y) for y in range(8,20)]
eye2_coordinates_pattern_x = ['eye_lmk_x_' + str(x) for x in range(36,48)]
eye2_coordinates_pattern_y = ['eye_lmk_y_' + str(y) for y in range(36,48)]

fig, axes = plt.subplots(8, 5, figsize=(25, 15))
axes = axes.flatten()

for i, videoName in enumerate(small_videoList):
    videoCsv = base_dir + videoName + '\\' + videoName + '.csv'
    df = pd.read_csv(videoCsv)
    df.columns = columns

    # Eye1
    eye1_x_points_df = df[eye1_coordinates_pattern_x]
    eye1_y_points_df = df[eye1_coordinates_pattern_y]
    eye1_area_for_all_frames = np.zeros(shape=len(df))

    # Eye2
    eye2_x_points_df = df[eye2_coordinates_pattern_x]
    eye2_y_points_df = df[eye2_coordinates_pattern_y]
    eye2_area_for_all_frames = np.zeros(shape=len(df))
    
    for j in range(len(df)):

        # Eye1
        eye1_x_points = eye1_x_points_df.iloc[j].to_numpy()
        eye1_y_points = eye1_y_points_df.iloc[j].to_numpy()
        eye1_area_frame_j = polyArea(eye1_x_points, eye1_y_points)
        eye1_area_for_all_frames[j] = eye1_area_frame_j

        # Eye2
        eye2_x_points = eye2_x_points_df.iloc[j].to_numpy()
        eye2_y_points = eye2_y_points_df.iloc[j].to_numpy()
        eye2_area_frame_j = polyArea(eye2_x_points, eye2_y_points)
        eye2_area_for_all_frames[j] = eye2_area_frame_j

    eyes_area_df = pd.DataFrame({'frame': df.frame, 'eye1_area': eye1_area_for_all_frames, 'eye2_area': eye2_area_for_all_frames})
    palette = sns.color_palette()
    sns.lineplot(x='frame', y='eye1_area', data=eyes_area_df, ax=axes[i])
    sns.lineplot(x='frame', y='eye2_area', data=eyes_area_df, ax=axes[i])
    axes[i].set(xlabel='Frame', ylabel='Area', title=videoName)
    axes[i].legend(['Eye_1 area', 'Eye_2 area'], loc='lower left')

plt.tight_layout()
plt.show()

# %% [markdown]
# #### Visualizzazione dell'intensità delle *action unit*, nel tempo, per un video preso casualmente.

# %%
import random as rd
randVideoName = videoList[rd.randint(0, len(videoList))]
randVideoCsv = base_dir + randVideoName + '\\' + randVideoName + '.csv'
df = pd.read_csv(randVideoCsv)
df.columns = columns
# Plot all Action Unit time series. 
au_regex_pat = re.compile(r'^AU[0-9]+_r$')
au_columns = df.columns[df.columns.str.contains(au_regex_pat)]
fig,axes = plt.subplots(6, 3, figsize=(10,12), sharex=True, sharey=True)
axes = axes.flatten()
for au_ix, au_col in enumerate(au_columns):
    sns.lineplot(x='frame', y=au_col, hue='face_id', data=df, ax=axes[au_ix])
    axes[au_ix].set(title=au_col, ylabel='Intensity')
    axes[au_ix].legend(loc=5)
plt.suptitle(randVideoName, y=1.02)
plt.tight_layout()

# %%
