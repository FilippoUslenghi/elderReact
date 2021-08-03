import os
import sys


model = sys.argv[1]
file = model + '.py'
emotions = ['anger', 'disgust', 'fear',
            'happiness', 'sadness', 'surprise', 'valence']
poses = ['tilted', 'frontal', 'none']

for emotion in emotions:
    for pose in poses:
        os.system(' '.join(['python', file, emotion, pose]))
        # print(' '.join([file, emotion, pose]))
