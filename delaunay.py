"""
Questo script estrae le feature relative alla triangolazione 
di Delaunay dei landamrk del volto estratti con OpenFace.
"""

import os
import re
import json
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay


def polygon_area(x, y, n):
    # Initialize area
    area = 0.0

    # Calculate value of Shoelace formula
    j = n - 1  # n is the number of points
    for i in range(0, n):
        area += (x[j] + x[i]) * (y[j] - y[i])  # (X[i], Y[i]) are coordinates of i'th point.
        j = i  # j is previous vertex to i

    # Return absolute value
    return abs(area // 2.0)

# Creation of the scheme of the triangles:

# frontal_face_df = pd.read_csv(os.path.join('faceMesh','frontal_face_smile_openface.csv'))
# frontal_face_df.columns = [column.replace(' ', '') for column in frontal_face_df.columns]
# landmarks_regex = re.compile(r'^x_[0-9]+$|^y_[0-9]+$')

# landmarks_locs = frontal_face_df.columns[frontal_face_df.columns.str.contains(landmarks_regex)]
# landmarks = frontal_face_df[landmarks_locs].iloc[0].to_numpy(dtype=float)
# landmarks_points = np.ndarray(shape=(68,2), dtype=float)

# for i in range(68): # for every landmarks
#     landmarks_points[i,0] = landmarks[i] # x coordinates
#     landmarks_points[i,1] = landmarks[68+i] # y coordinates

# tri = Delaunay(landmarks_points) # compute Delaunay triangulation
# triangles = landmarks_points[tri.simplices]

# data = {TRIANGLES_SCHEME: triangles}
# with open('triangles_scheme.json', 'w') as f:
#     json.dump(data, f)

# loading the triangle scheme
with open("triangles_scheme.json") as jsonFile:
    jsonObject = json.load(jsonFile)
    jsonFile.close()

TRIANGLES_SCHEME = jsonObject.get('TRIANGLES_SCHEME')
datasets = ['train', 'dev', 'test']
for dataset in datasets:

    base_dir = os.path.join('openFace', dataset, 'processed_interpolated', '')
    landmarks_regex = re.compile(r'^x_[0-9]+$|^y_[0-9]+$')
    delaunay_df_columns = ['frame']  # initialize the columns of the dataframe
    delaunay_df_columns.extend(
        [f'triangle_{i}' for i in range(len(TRIANGLES_SCHEME))]
    )  # initialize the columns of the dataframe

    flag = True
    for csv in os.listdir(base_dir):  # for every video

        openface_df = pd.read_csv(os.path.join(base_dir, csv))
        delaunay_df_data = []  # initialize the data of the dataframe

        if flag:
            landmarks_locs = openface_df.columns[openface_df.columns.str.contains(landmarks_regex)]
            flag = False

        starting_frame = openface_df.frame.iloc[0]
        for index in openface_df.index:  # for every frame
            delaunay_df_row_values = [index + starting_frame]  # initialize the data of the row, in this case, the value of the 'frame' column
            landmarks = openface_df[landmarks_locs].iloc[index].to_numpy(dtype=float)
            landmarks_points = np.ndarray(shape=(68, 2), dtype=float)

            for i in range(68):  # for every landmarks
                landmarks_points[i, 0] = landmarks[i]  # x coordinates
                landmarks_points[i, 1] = landmarks[68 + i]  # y coordinates

            triangles = landmarks_points[TRIANGLES_SCHEME]
            areas = []
            for i, triangle in enumerate(triangles):
                X_coords = triangle[:, 0]
                Y_coords = triangle[:, 1]
                area = polygon_area(X_coords, Y_coords, 3)
                areas.append(area)

            tot_area = sum(areas)
            normalized_aras = np.divide(areas, tot_area)

            # builds the dataframe
            delaunay_df_row_values.extend(normalized_aras)  # build the row
            delaunay_df_row = dict(zip(delaunay_df_columns, delaunay_df_row_values))  # link the value to the column
            delaunay_df_data.append(delaunay_df_row)  # append the row

        delaunay_df = pd.DataFrame(data=delaunay_df_data)  # create the daframe with the data gathered

        os.makedirs(os.path.join('openFace', dataset, 'delaunay'), exist_ok=True)
        out_dir = os.path.join('openFace', dataset, 'delaunay')
        delaunay_df.to_csv(os.path.join(out_dir, f'{csv}.csv'), index=False)
