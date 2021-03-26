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
# Plotto la confidence nel tempo di ogni video e salvo l'immagine nella relativa direcorty

# %%
base_dir = 'C:\\Users\\us98\\PycharmProjects\\elderReactProject\\myProcessed\\'
videoList = os.listdir(base_dir)

for videoName in videoList:
    videoCsv = base_dir + videoName + '\\' + videoName + ".csv"
    df = pd.read_csv(videoCsv)
    df.columns = [col.replace(" ", "") for col in df.columns]
    
    #commento perché plottare fa perdere tempo
    # cf = df["confidence"]
    # time = df["frame"]
    # plt.figure()
    # plt.title('Confidence throughout the video "' + videoName + '.avi"')
    # plt.xlabel("Frame number")
    # plt.ylabel("Confidence")
    # plt.plot(time, cf)
    # plt.yticks([x/10 for x in range(10)])
    # # plt.savefig(f"{videoName}_confidence.jpg", format="jpg", dpi=150)
    # plt.show()

# %% [markdown]
# ##Analisi della confidence
# Voglio utilizzare lo zero crossing rate per analizzare la confindence nei video.
# Per fare questo sottraggo al valore della confidence 0.75 in modo tale da avere uno zero corssing ogni volta
# che la confidence scende (e risale) da tale valore.
# Faccio poi un plot dello zero crossing rate per ogni video
# Se un video ha un alto zero crossing rate allora non vi è una continua affidabilità dei risultati.

# %%
zeroCrossing_values = []

csv_dir = 'C:\\Users\\us98\\PycharmProjects\\elderReactProject\\myProcessed_csv\\'

csv_files = os.listdir(csv_dir)
for csv_file in csv_files:
    df = pd.read_csv(csv_dir + csv_file)
    df.columns = [col.replace(" ", "") for col in df.columns]

    cf = np.array(df.confidence - 0.75)
    number_of_zeroCrosses = np.diff(cf > 0).sum()

    zeroCrossing_values.append(number_of_zeroCrosses)

data = {
    'videos': csv_files,
    'zeroCrossingRate': zeroCrossing_values
}

ZCR_df = pd.DataFrame(data, columns=['videos', 'zeroCrossingRate'])

plt.bar(ZCR_df.index, ZCR_df.zeroCrossingRate)
plt.title("Zero crossing rate of the confidence for each video")
plt.xlabel("Video's index")
plt.ylabel("ZCR")
plt.yticks([x for x in range(100)][::10])
plt.show()

print("Medium ZCR ", ZCR_df.zeroCrossingRate.mean())

# %% [markdown]
# Andando a guardare i video di openFace che hanno uno ZCR=16
# si nota che la maggior parte riguardano la stessa persona.
# Questo probabilmente è dovuto al fatto che indossando gli occhiali
# il tool non riesce a mantenere una condifence alta.

# %%
print(ZCR_df.videos[ZCR_df.zeroCrossingRate == 16])

# %% [markdown]
# Vediamo quanti video hanno un ZCR>=10 e osserviamone i grafici della confidence

# %%
videos = ZCR_df.videos[ZCR_df.zeroCrossingRate >= 10].tolist()
print("Numero di video ", len(videos))

plots = [x.replace(".csv", "_confidence.jpg") for x in videos]
base_dir= "C:\\Users\\us98\\PycharmProjects\\elderReactProject\\myProcessed"


plt.figure(figsize=(150,100))
for i, plot in enumerate(plots):
    if i == 0:
        first_img = cv2.imread(base_dir + '\\' + plot[:-15] + '\\' + plot)
        ax1 = plt.subplot(8, 6, i+1)
        plt.imshow(first_img)
        continue
    img = cv2.imread(base_dir + '\\' + plot[:-15] + '\\' + plot)
    plt.subplot(8, 6, i+1, sharex=ax1, sharey=ax1)
    plt.imshow(img)
plt.tight_layout()

# %% [markdown]
# #Visualizzazione dati
# Procedo con la visualizzazione delle features di 40 video presi casualmente

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
    axes[i].set(ylim=[-1.5, 1.5], xlabel='Frame Number', ylabel="Radians")
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

# %%
