import os
import sys


# model, features = sys.argv[1], sys.argv[2]
script = sys.argv[1]
features_list = ['delaunay', 'au_intensities', 'au_activations', 'au_intensities_activations']
emotions = ['anger', 'disgust', 'fear',
            'happiness', 'sadness', 'surprise', 'valence']
poses = ['tilted', 'frontal', 'none']
# au_features = ['au_intensities', 'au_activations', 'au_intensities_activations']

for features in features_list:
    
    for emotion in emotions:

        if script == 'lstm.py':
            if features == 'delaunay': continue
            os.system(' '.join(['python', script, emotion, features]))
            # print(' '.join(['python', script, emotion, features]))

        else:
            for pose in poses:
                os.system(' '.join(['python', script, emotion, pose, features]))
                # print(' '.join([script, emotion, pose, features]))
