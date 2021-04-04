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
videoList = os.listdir(base_dir)
small_videoList = videoList[::15][:-1]
columns = [col.replace(" ", "") for col in pd.read_csv(base_dir + '50_50_4\\50_50_4.csv').columns]

# %% [markdown]
# # Confidence
# Per ogni video voglio vedere che valore assume la varianza della confidence.
# Questo perhé sono interessato ad imporre una threshold su di essa in base alla quale selezionare i video che manterrò.
# %%
print("Varianza della confidence per ogni video:")

plt.figure(figsize=(10, 8))

variance_list = []
for videoName in small_videoList:
    videoCsv = base_dir + videoName + '\\' + videoName + ".csv"
    df = pd.read_csv(videoCsv)
    df.columns = columns
    
    variance_list.append(df.confidence.var())

variance_df = pd.DataFrame({'video': small_videoList, 'variance': variance_list})

plt.barh(variance_df.video, variance_df.variance)
plt.title('Variance trend in a small set of the dataset')
plt.xlabel='Videos'
plt.ylabel='Variance'
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Selezione sui video di small_videoList

# %%
print('Confidence dei video selezionati in base alla threshold')
threshold = variance_df.mean()
new_small_videoList = variance_df[variance_df.variance<threshold[0]]

fig, axes = plt.subplots(8, 5, figsize=(25, 20), sharey=True)
axes = axes.flatten()

for i, videoName in enumerate(new_small_videoList.video):
    videoCsv = base_dir + videoName + '\\' + videoName + ".csv"
    df = pd.read_csv(videoCsv)
    df.columns = columns

    df[['confidence']].plot(ax=axes[i], legend = False)
    axes[i].set(xlabel='Frame Number', ylabel="Confidence", title=videoName)

plt.yticks([x/10 for x in range(11)])
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Selezione sui video di videoList

# %%
print('Confidence dei video selezionati in base alla threshold')
threshold = variance_df.mean()
new_small_videoList = variance_df[variance_df.variance<threshold[0]]

fig, axes = plt.subplots(62, 10, figsize=(25, 20), sharey=True)
axes = axes.flatten()

for i, videoName in enumerate(videoList):
    videoCsv = base_dir + videoName + '\\' + videoName + ".csv"
    df = pd.read_csv(videoCsv)
    df.columns = columns

    df[['confidence']].plot(ax=axes[i], legend = False)
    axes[i].set(xlabel='Frame Number', ylabel="Confidence", title=videoName)

plt.yticks([x/10 for x in range(11)])
plt.tight_layout()
plt.show()

# %%
