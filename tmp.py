# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
# Plotto la confidence nel tempo di ogni video e salvo l'immagine nella relativa direcorty

# %%
base_dir = 'C:\\Users\\us98\\PycharmProjects\\elderReactProject\\myProcessed\\'
videoList = os.listdir(base_dir)

for videoName in videoList:
    csvVideo = base_dir + videoName + '\\' + videoName + ".csv"
    df = pd.read_csv(csvVideo)
    df.columns = [col.lstrip().rstrip() for col in df.columns]
    
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
import cv2

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
# Procedo con la visualizzazione delle features di 40 video presi casualmente

# %% [markdown]
# Visualizzazione del gaze angle

# %%
base_dir = 'C:\\Users\\us98\\PycharmProjects\\elderReactProject\\myProcessed\\'
small_videoList = videoList[::15][:-1]

fig, axes = plt.subplots(8, 5, figsize=(25, 20), sharey=True)
axes = axes.flatten()

for i, videoName in enumerate(small_videoList):
    csvVideo = base_dir + videoName + '\\' + videoName + '.csv'
    df = pd.read_csv(csvVideo)
    df.columns = [col.lstrip().rstrip() for col in df.columns]

    df[['gaze_angle_x', 'gaze_angle_y']].plot(ax=axes[i], legend = False)
    axes[i].set(ylim=[-1.5, 1.5], xlabel='Frame Number', ylabel="Radians")
    axes[i].legend(['gaze_angle_x', 'gaze_angle_y'], loc='lower left')

plt.tight_layout()
plt.show()

# %% [markdown]
# Visualizzazione delle coordinate 3D del *gaze vector*

# %%
from mpl_toolkits.mplot3d import Axes3D

base_dir = 'C:\\Users\\us98\\PycharmProjects\\elderReactProject\\myProcessed\\'

fig = plt.figure(figsize=(25, 10))
for i, videoName in enumerate(small_videoList):
    csvVideo = base_dir + videoName + '\\' + videoName + '.csv'
    df = pd.read_csv(csvVideo)
    df.columns = [col.replace(" ", "") for col in df.columns]


    # Plot delle coordinate spaziali del gaze vector
    ax = fig.add_subplot(4, 10, i+1, projection='3d')

    ax.plot(df.gaze_0_x, df.gaze_0_y, df.gaze_0_z, color='blue')
    ax.plot(df.gaze_1_x, df.gaze_1_y, df.gaze_1_z, color='red')
    ax.set_title(videoName)
    ax.set(xlabel='x', ylabel='y', zlabel='z', xticks=[], yticks=[], zticks=[])
    ax.set_xlabel(xlabel='x', labelpad=0)
    ax.set_ylabel(ylabel='y', labelpad=0)
    ax.set_zlabel(zlabel='z', labelpad=0)
    ax.legend(['Leftmost eye', 'Rightmost eye'], fontsize='xx-small')

plt.tight_layout()
plt.show()
# %%
