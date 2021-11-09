"""
Script per eseguire l'addestramento dei modelli di machine learning.
"""

import os

scripts = ['dummy_classifier.py', 'logistic_regression.py', 'linear_SVM.py', 'kernel_SVM.py', 'lstm.py', 'lstm_delaunay.py']
features_list = ['delaunay', 'au_intensities', 'au_activations', 'au_intensities_activations']
emotions = ['anger', 'disgust', 'fear',
            'happiness', 'sadness', 'surprise', 'valence']
poses = ['tilted', 'frontal', 'none']

for script in scripts:
    for features in features_list: 
        for emotion in emotions:

            if script == 'lstm.py':
                if features == 'delaunay': continue
                os.system(' '.join(['python3', script, emotion, features]))

            elif script == 'lstm_delaunay.py':
                if features != 'delaunay': continue
                os.system(' '.join(['python3', script, emotion, features]))

            else:
                for pose in poses:
                    os.system(' '.join(['python3', script, emotion, pose, features]))
