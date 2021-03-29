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

# for videoName in videoList:
#     videoCsv = base_dir + videoName + '\\' + videoName + ".csv"
#     df = pd.read_csv(videoCsv)
#     df.columns = [col.replace(" ", "") for col in df.columns]

    
#     cf = df["confidence"]
#     time = df["frame"]
#     plt.figure()
#     plt.title('Confidence throughout the video "' + videoName + '.avi"')
#     plt.xlabel("Frame number")
#     plt.ylabel("Confidence")
#     plt.plot(time, cf)
#     plt.yticks([x/10 for x in range(10)])
#     plt.savefig(f"{videoName}_confidence.jpg", format="jpg", dpi=150)
#     plt.show()

# %% [markdown]
# ## Analisi della confidence
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
# Vediamo ora il grafico di alcuni video con ZCR<10

# %%
videos = ZCR_df.videos[ZCR_df.zeroCrossingRate < 10].tolist()[:48]
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
