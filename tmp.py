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

i=1
plt.figure(figsize=(150,100))
for plot in plots:
    if i == 1:
        first_img = cv2.imread(base_dir + '\\' + plot[:-15] + '\\' + plot)
        ax1 = plt.subplot(8, 6, i)
        plt.imshow(first_img)
        i+=1
        continue
    img = cv2.imread(base_dir + '\\' + plot[:-15] + '\\' + plot)
    plt.subplot(8, 6, i, sharex=ax1, sharey=ax1)
    plt.imshow(img)
    i+=1
plt.tight_layout()

# %% [markdown]
# Procedo con la visualizzazione delle features di 40 video presi casualmente

# %%
base_dir = 'C:\\Users\\us98\\PycharmProjects\\elderReactProject\\myProcessed\\'
small_videoList = videoList[::15][:-1]

stop = 0
for videoName in small_videoList:
    if stop == 1: break
    csvVideo = base_dir + videoName + '\\' + videoName + '.csv'
    df = pd.read_csv(csvVideo)
    df.columns = [col.lstrip().rstrip() for col in df.columns]

    gazeAng_x = df["gaze_angle_x"]
    gazeAng_y = df["gaze_angle_y"]
    time = df["frame"]


    f, axes = plt.subplots(2, len(df), figsize=(10, 5))
    for faces_ix, face_id in enumerate(df[::-1]):
        df.plot.scatter(x='gaze_angle_x', y='gaze_angle_y', ax=axes[0])
        axes[0].scatter(0, 0, marker='x', color='k')  # draw origin.
        axes[0].set(xlim=[-2, 2], ylim=[-2, 2], title=f'Gaze movement of face_id=={face_id}')
        df[['gaze_angle_x', 'gaze_angle_y']].plot(ax=axes[1])
        axes[1].set(ylim=[-1.5, 1.5], xlabel='Frame Number', ylabel="Radians")
    plt.tight_layout()
    plt.show()

    stop += 1


    # plt.plot(time, gazeAng_x, color="blue")
    # plt.plot(time, gazeAng_y, color="red")
    # plt.title('Gaze angle in video "' + videoName + '"' )
    # plt.xlabel('Time')
    # plt.ylabel('Gaze angle')
    # plt.show()

    # time = df["frame"]
    # plt.figure()
    # plt.title('Confidence throughout the video "' + videoName + '.avi"')
    # plt.xlabel("Frame number")
    # plt.ylabel("Confidence")
    # plt.plot(time, cf)
    # plt.yticks([x/10 for x in range(10)])
    # # plt.savefig(f"{videoName}_confidence.jpg", format="jpg", dpi=150)
    # plt.show()



# %%


# %%
