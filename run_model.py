import os
import sys


model, features = sys.argv[1], sys.argv[2]
file = model + '.py'
emotions = ['anger', 'disgust', 'fear',
            'happiness', 'sadness', 'surprise', 'valence']
poses = ['tilted', 'frontal', 'none']

for emotion in emotions:

    if model == 'lstm':
        os.system(' '.join(['python', file, emotion, features]))
        # print(' '.join(['python', file, emotion, features]))
    else:
        for pose in poses:
            # os.system(' '.join(['python', file, emotion, pose, features]))
            print(' '.join([file, emotion, pose, features]))
