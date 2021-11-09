"""
Questo script ricrea le annotazioni del dataset eliminando quelle relative ai video scartati
"""

import os

groups = ['train', 'dev', 'test']
for group in groups:

    annotation_file = f'{group}_labels.txt'
    old_annotations_path = os.path.join('dataset', 'ElderReact-master', 'Annotations')
    new_annotations_path = os.path.join('my_dataset', 'Annotations')
    os.makedirs(new_annotations_path, exist_ok=True)
    line_list = []
    with open(os.path.join(old_annotations_path, annotation_file)) as f:
        for line in f.readlines():
            video_name = line.split(sep=' ')[0].replace('.mp4', '.csv')
            if video_name in os.listdir(os.path.join('my_dataset', 'Features', group, 'interpolated_AU_')):
                line_list.append(line)
                
        try:
            with open(os.path.join(new_annotations_path, annotation_file), 'x') as new_f:
                new_f.write(''.join([line for line in line_list]))
        except FileExistsError:
            pass
