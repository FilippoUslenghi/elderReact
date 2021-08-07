import os
import sys


model, features = sys.argv[1], sys.argv[2]
file = model + '.py'
emotions = ['anger', 'disgust', 'fear',
            'happiness', 'sadness', 'surprise', 'valence']
poses = ['tilted', 'frontal', 'none']
au_features = ['au_intensities', 'au_activations', 'au_intensities_activations']

for emotion in emotions:

    if model == 'lstm':
        os.system(' '.join(['python', file, emotion, features]))
        # print(' '.join(['python', file, emotion, features]))
    else:
        if features == 'au':
            for pose in poses:
                for au_feature in au_features:
                    os.system(' '.join(['python', file, emotion, pose, au_feature]))
                    # print(' '.join([file, emotion, pose, au_feature]))    
        else:
            for pose in poses:
                os.system(' '.join(['python', file, emotion, pose, features]))
                # print(' '.join([file, emotion, pose, features]))
