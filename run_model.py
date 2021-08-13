import os
import sys


script = sys.argv[1]
features_list = ['delaunay', 'au_intensities', 'au_activations', 'au_intensities_activations']
emotions = ['anger', 'disgust', 'fear',
            'happiness', 'sadness', 'surprise', 'valence']
poses = ['tilted', 'frontal', 'none']

for features in features_list:
    
    for emotion in emotions:

        if script == 'lstm.py':
            if features == 'delaunay': continue
            os.system(' '.join(['python', script, emotion, features]))
            # print(' '.join(['python', script, emotion, features]))

        elif script == 'kernel_SVM.py':
            # if features == 'delaunay': continue
            if features[:2] == 'au': continue
            for pose in poses:
                os.system(' '.join(['python', script, emotion, pose, features]))

        else:
            for pose in poses:
                os.system(' '.join(['python', script, emotion, pose, features]))
                # print(' '.join([script, emotion, pose, features]))


# script, features = sys.argv[1], sys.argv[2]
# emotions = ['anger', 'disgust', 'fear',
#             'happiness', 'sadness', 'surprise', 'valence']
# poses = ['tilted', 'frontal', 'none']
# au_features = ['au_intensities', 'au_activations', 'au_intensities_activations']
    
# for emotion in emotions:

#     if script == 'lstm.py':
#         if features == 'au':
#             for au_feature in au_features:
#                 os.system(' '.join(['python', script, emotion, au_feature]))
#                 # print(' '.join(['python', script, emotion, au_feature]))

#     else:
#         for pose in poses:
#             os.system(' '.join(['python', script, emotion, pose, features]))
#             # print(' '.join([script, emotion, pose, features]))