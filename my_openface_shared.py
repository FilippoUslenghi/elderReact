# %% [markdown]
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# <a href="https://colab.research.google.com/gist/jcheong0428/c16146b386ea60fab888b56e8e5ee747/openface_shared.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# %% [markdown]
# Facial Feature Detection with OpenFace
# 
# This notebook uses an open source project [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace) by Tadas Baltrusaitis to detect and track single/multi-person head motions and facial muscle movements on a given Youtube video. This notebook was inspired by [DL-CoLab-Notebooks](https://github.com/tugstugi/dl-colab-notebooks).
# %% [markdown]
# Instead of `FaceLandmarkVidMulti` you may also use `FeatureExtraction` to extract features of a single face or `FaceLandmarkImg` to extract 3333features on a face image. See full description of the arguments [here](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Command-line-arguments). 

# %%
#import cv2
import numpy as np 
import matplotlib.pyplot as plt
import os 
import time
from ipywidgets import interact
import ipywidgets as widgets
import random

from pandas.core.frame import DataFrame


# %%
base_dir = os.path.join(os.getcwd(), 'dataset\\ElderReact_Data\\ElderReact_train\\')


for filename in os.listdir(base_dir):

    FILENAME = base_dir + filename
#    os.system(os.getcwd() + '\\OpenFace\\FeatureExtraction.exe -f ' +  FILENAME + ' -out_dir myProcessed\\' + filename[:-4])

# %% [markdown]
#Scelgo un file ogni 5 video; così se ho, per esempio, 15 video di reazioni ad uno stimolo, avrò 3 file rappresentativi
#mentre se ho 6 video relativi ad un altro stimolo ne avrò 1 ecc ecc.

# %%
filenames = []
for i, filename in enumerate(os.listdir("myProcessed")):
    if i%5==0:
        filenames.append(filename)
print("Numero di video scelti: ", len(filenames))

# %% [markdown]
# # Extra: Here are some tips for loading and plotting the data. 

# %%
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
import pandas as pd, seaborn as sns
sns.set_style('white')  
import matplotlib.pyplot as plt 

# Load data
processed_csvs = []
for filename in filenames:
    df = pd.read_csv('myProcessed/'+ filename + '/' + filename + '.csv')
    processed_csvs.append(df)
    # Remove empty spaces in column names.
    df.columns = [col.replace(" ", "") for col in df.columns]
    # Print few values of data.
    # print(f"Max number of frames {df.frame.max()}", f"\nTotal shape of dataframe {df.shape}")
processed_csvs[-1].head()


# %% [markdown]
#Visualizza un dataframe scelto a caso

# %%
# df = processed_csvs[random.randint(0, len(processed_csvs))]
df = processed_csvs[-1]

# %%
# See how many unique faces there are
print("Number of unique faces: ", len(df.face_id.unique()), "\nList of face_id's: ", df.face_id.unique())

# %%
df.groupby('face_id').mean()['confidence']

# %%
import re
x_regex_pat = re.compile(r'^x_[0-9]+$')
y_regex_pat = re.compile(r'^y_[0-9]+$')
x_locs = df.columns[df.columns.str.contains(x_regex_pat)]
y_locs = df.columns[df.columns.str.contains(y_regex_pat)]

no_unique_faces = len(df.face_id.unique())
palette = sns.color_palette()  

avg_face_df = pd.DataFrame({'x_locs':df[x_locs].mean(axis=1), 'y_locs':df[y_locs].mean(axis=1), 'face_id': df.face_id})
ax = sns.scatterplot(x='x_locs', y='y_locs', hue = 'face_id', data=avg_face_df, marker="+")#, palette=palette)
ax.set(xlim=[0, 1920], ylim=[1080,0], title="Before thresholding");

# %%
avg_face_df_conf = avg_face_df[df.confidence>=.80]
no_unique_faces = len(avg_face_df_conf.face_id.unique())
ax = sns.scatterplot(x='x_locs', y='y_locs', hue = 'face_id', data=avg_face_df_conf, marker="+", palette=palette[:no_unique_faces])
ax.set(xlim=[0, 1920], ylim=[1080,0], title="After thresholding");

# %% [markdown]
# Let's clean our data with a threshold of 80% confidence and plot the AU trajectories for all AUs.
# 

# %%
# Threshold data by 80%
plt.style.use('seaborn')
df_clean = df[df.confidence>=.80]
# Plot all Action Unit time series. 
au_regex_pat = re.compile(r'^AU[0-9]+_r$')
au_columns = df.columns[df.columns.str.contains(au_regex_pat)]
print("List of AU columns:", au_columns)
f,axes = plt.subplots(6, 3, figsize=(10,12), sharex=True, sharey=True)
axes = axes.flatten()
for au_ix, au_col in enumerate(au_columns):
    sns.lineplot(x='frame', y=au_col, hue='face_id', data=df_clean, ax=axes[au_ix])
    axes[au_ix].set(title=au_col, ylabel='Intensity')
    axes[au_ix].legend(loc=5)
plt.suptitle("AU intensity predictions by time for each face", y=1.02)
plt.tight_layout()

# %% [markdown]
# We could also compare how synchronized each individuals are to one another during the interaction by using a simple Pearson correlation.

# %%
# Let's compare how much AU12 (smiling) activity occurs at similar times across people.
df_clean.pivot(index='frame', columns='face_id', values='AU12_r').corr()

# %% [markdown]
# # Lastly, here is just a few lines of code to get you started on working with gaze directions. 

# %%
f,axes = plt.subplots(2,len(df_clean.face_id.unique()), figsize=(10,5))
for faces_ix, face_id in enumerate(df_clean.face_id.unique()[::-1]):
  df_clean.query(f'face_id=={face_id}').plot.scatter(x='gaze_angle_x', y='gaze_angle_y', ax=axes[0][faces_ix])
  axes[0][faces_ix].scatter(0,0, marker='x', color = 'k') # draw origin.
  axes[0][faces_ix].set(xlim=[-2,2], ylim=[-2,2], title=f'Gaze movement of face_id=={face_id}')
  df_clean.query(f'face_id=={face_id}')[['gaze_angle_x', 'gaze_angle_y']].plot(ax=axes[1][faces_ix])
  axes[1][faces_ix].set(ylim=[-1.5,1.5], xlabel='Frame Number', ylabel="Radians")
plt.tight_layout()
plt.show()

# %% [markdown]
# # That's it for now. Hope you enjoyed this tutorial.
# 
# ## Additional resources
# *   [OpenFace Github Page](https://github.com/TadasBaltrusaitis/OpenFace)
# *   [Medium article on more ways to assess synchrony in time series data](https://towardsdatascience.com/four-ways-to-quantify-synchrony-between-time-series-data-b99136c4a9c9)
# *   [Comparison of facial emotion recognition software: OpenFace vs Affectiva vs FACET](https://medium.com/@jinhyuncheong/face-analysis-software-comparison-affectiva-affdex-vs-openface-vs-emotient-facet-5f91a4f12cbb)
# 
# *This notebook was prepared by [Jin Hyun Cheong](http://jinhyuncheong.com).*

# %%

