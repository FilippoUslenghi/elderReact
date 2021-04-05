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
some_videos = all_videos[::15][:-1]
columns = [col.replace(" ", "") for col in pd.read_csv(base_dir + '50_50_4\\50_50_4.csv').columns]

# %% [markdown]
# # Confidence
# Per ogni video voglio vedere che valore assume la varianza della confidence.
# Sono poi interessato ad imporre una threshold su di essa in base alla quale selezionare i video che manterrò.
# %%
print("Varianza della confidence per ogni video:")

plt.figure(figsize=(10, 8))

some_videos_var_list = []
for videoName in some_videos:
    videoCsv = base_dir + videoName + '\\' + videoName + ".csv"
    df = pd.read_csv(videoCsv)
    df.columns = columns
    
    some_videos_var_list.append(df.confidence.var())

some_videos_var_df = pd.DataFrame({'video': some_videos, 'variance': some_videos_var_list})

plt.barh(some_videos_var_df.video, some_videos_var_df.variance)
plt.title('Variance trend in a small set of the dataset')
plt.xlabel='Videos'
plt.ylabel='Variance'
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Inizializzo una threshold

# %%
threshold = some_videos_var_df.mean()[0]
print('Threshold: ', threshold)

# %% [markdown]
# Applico la threshold su un gruppo di 40 video presi casualmente

# %%
print('Confidence dei video selezionati in base alla threshold...')
new_some_videos = some_videos_var_df[some_videos_var_df.variance<threshold]

fig, axes = plt.subplots(8, 5, figsize=(25, 20), sharey=True)
axes = axes.flatten()

for i, videoName in enumerate(new_some_videos.video):
    videoCsv = base_dir + videoName + '\\' + videoName + ".csv"
    df = pd.read_csv(videoCsv)
    df.columns = columns

    df[['confidence']].plot(ax=axes[i], legend = False)
    axes[i].set(xlabel='Frame Number', ylabel="Confidence", title=videoName)

plt.ylim(0,1)
plt.yticks([x/10 for x in range(11)])
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Creazione del subset di video

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
# ###### Osserviamo l'andamento della confidence di un sottoinsieme dei video con varianza superiore alla media

# %%
videos_above_avg = good_videos_var_df[good_videos_var_df.variance>=good_videos_var_df.variance.mean()]
n_videos_above_avg = len(videos_above_avg)
print('Numero di video con varianza sopra la media:', n_videos_above_avg)

fig, axes = plt.subplots(8, 5, figsize=(25, 20), sharey=True)
axes = axes.flatten()

for i, videoName in enumerate(videos_above_avg.video[:40]):
    videoCsv = base_dir + videoName + '\\' + videoName + ".csv"
    df = pd.read_csv(videoCsv)
    df.columns = columns

    df[['confidence']].plot(ax=axes[i], legend = False)
    axes[i].set(xlabel='Frame Number', ylabel="Confidence", title=videoName)

plt.ylim(0,1)
plt.yticks([x/10 for x in range(11)])
plt.tight_layout()
plt.show()

# %% [markdown]
# ###### Ora nel caso di varianza inferiore alla media

# %%
videos_below_avg = good_videos_var_df[good_videos_var_df.variance<good_videos_var_df.variance.mean()]
n_videos_below_avg = len(videos_below_avg)
print('Numero di video con varianza sotto la media:', n_videos_below_avg)

fig, axes = plt.subplots(8, 5, figsize=(25, 20), sharey=True)
axes = axes.flatten()

for i,videoName in enumerate(videos_below_avg.video[:40]):
    videoCsv = base_dir + videoName + '\\' + videoName + ".csv"
    df = pd.read_csv(videoCsv)
    df.columns = columns
    
    df[['confidence']].plot(ax=axes[i], legend = False)
    axes[i].set(xlabel='Frame Number', ylabel="Confidence", title=videoName)

plt.ylim(0,1)
plt.yticks([x/10 for x in range(11)])
plt.tight_layout()
plt.show()

# %%
