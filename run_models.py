import os


files = ['dummy_classifier.py', 'logistic_regression.py']
emotions = ['anger', 'disgust', 'fear',
            'happiness', 'sadness', 'surprise', 'valence']
poses = ['tilted', 'frontal', 'none']

for file in files:
    for emotion in emotions:
        for pose in poses:
            os.system(' '.join(['python', file, emotion, pose]))
            # print(' '.join([file, emotion, pose]))
