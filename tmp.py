# %%
import os
import numpy as np
from numpy.lib.function_base import disp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
os.chdir("myProcessed_csv")

# %%
filenames = os.listdir()[::20]
for filename in filenames:
    df = pd.read_csv(filename)
    df.columns = [col.lstrip().rstrip() for col in df.columns]
    

    cf = df["confidence"]
    time = df["frame"]
    plt.figure()
    plt.plot(time, cf)
    plt.yticks([x/10 for x in range(10)])
    plt.show()
    print(filename, "length: ", len(time))

# %% [markdown]
# Plotto la confidence nel tempo di ogni video e salvo l'immagine nella relativa direcorty

# %%
filenames = os.listdir("C:\\Users\\us98\\PycharmProjects\\elderReactProject\\myProcessed")

# for filename in filenames:
#     os.chdir(filename)
#     FILENAME = filename + ".csv"
#     df = pd.read_csv(FILENAME)
#     df.columns = [col.lstrip().rstrip() for col in df.columns]
    

#     cf = df["confidence"]
#     time = df["frame"]
#     plt.figure()
#     plt.title('Confidence throughout the video "' + filename + '.avi"')
#     plt.xlabel("Frame number")
#     plt.ylabel("Confidence")
#     plt.plot(time, cf)
#     plt.yticks([x/10 for x in range(10)])
#     plt.savefig(f"{filename}_confidence.jpg", format="jpg", dpi=150)
#     plt.show()
#     os.chdir('..')

# %% [markdown]
# Creo dei subpplot della confidence di alcuni video scelti ad intervalli arbitari

# %%
tot_cf = []
tot_frame = []

filenames = os.listdir()[::20]
for filename in filenames:
    df = pd.read_csv(filename)
    df.columns = [col.replace(" ", "") for col in df.columns]

    tot_cf.extend([df["confidence"]])
    tot_frame.extend([df["frame"]])

small_df = [tot_cf, tot_frame]
f, axes = plt.subplots(4, 8, figsize=(10, 12), sharex=True, sharey=True)
axes = axes.flatten()

for cf_ix, cf_col in enumerate(tot_cf):
	sns.lineplot(x='frame', y=cf_col,
                 data=small_df, ax=axes[cf_ix])
axes[cf_ix].set(title=cf_col, ylabel='Intensity')
axes[cf_ix].legend(loc=5)


# plt.figure()
# plt.plot(time, cf)
# plt.yticks(list(dict.fromkeys(df.confidence)))
# plt.show()
# print(filename, "length: ", len(time))

# %%
# Voglio utilizzare il zero crossing rate per analizzare la confindence nei video.
# Per fare questo sottraggo al valore della confidence 0.75 in modo tale da avere uno zero corssing ogni volta
# che la confidence scende (e risale) da tale valore.
# Faccio poi un plot dello zero crossing/2 rate per ogni video
# Se un video ha un alto zero crossing rate allora non vi è una continua affidabilità dei risultati.

# %%
zeroCrossing_values = []

csv_dir = 'C:\\Users\\us98\\PycharmProjects\\elderReactProject\\myProcessed_csv\\'

filenames = os.listdir(csv_dir)
for filename in filenames:
    df = pd.read_csv(csv_dir + filename)
    df.columns = [col.replace(" ", "") for col in df.columns]

    cf = np.array(df.confidence - 0.75)
    zero_crosses = np.diff(cf > 0).sum()

    zeroCrossing_values.append(zero_crosses/2)

data = {
    'videos': filenames,
    'zeroCrossingRate': zeroCrossing_values
}

zeroCrossing_df = pd.DataFrame(data, columns=['videos', 'zeroCrossingRate'])

plt.bar(zeroCrossing_df.index, zeroCrossing_df.zeroCrossingRate)
plt.title("Zero corssing rate for the shifted confidence")
plt.xlabel("Videos")
plt.ylabel("ZCR")
plt.yticks([0,10,20,30,40,50])
plt.show()
    
# %%
