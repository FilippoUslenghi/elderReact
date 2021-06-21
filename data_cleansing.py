import os
import numpy as np
import pandas as pd

def remove_first_frames(df: pd.DataFrame):
    LANDMARKS = [f'x_{i}' for i in range(68)] + [f'y_{i}' for i in range(68)]
    
    firstFrame_x = 0
    firstJumps_x = np.absolute(np.diff(df.x_27.values[:30]))>30
    if any(firstJumps_x):
        # finding the index of the last occurence
        firstJumps_x = firstJumps_x.tolist()
        firstJumps_x.reverse()
        firstFrame_x = len(firstJumps_x)-firstJumps_x.index(True)
        
    firstFrame_y = 0
    firstJumps_y = np.absolute(np.diff(df.y_27.values[:30]))>30
    if any(firstJumps_y):
        # finding the index of the last occurence
        firstJumps_y = firstJumps_y.tolist()
        firstJumps_y.reverse()
        firstFrame_y = len(firstJumps_y)-firstJumps_y.index(True)
        
    firstFrame = firstFrame_x if firstFrame_x>firstFrame_y else firstFrame_y
    if firstFrame!=0: df=df.iloc[firstFrame:]

    return df, int(firstFrame)


def remove_last_frames(df: pd.DataFrame):
    
    lastFrame_x = df.iloc[-1][0]
    lastJumps_x = np.absolute(np.diff(df.x_27.values[-30:]))>30
    
    if any(lastJumps_x):
        lastJumps_x = lastJumps_x.tolist()
        lastFrame_x = -(len(lastJumps_x)-lastJumps_x.index(True))
        
        
    lastFrame_y = df.iloc[-1][0]
    lastJumps_y = np.absolute(np.diff(df.y_27.values[-30:]))>30
        
    if any(lastJumps_y):
        lastJumps_y = lastJumps_y.tolist()
        lastFrame_y = -(len(lastJumps_y)-lastJumps_y.index(True))
    
    lastFrame = lastFrame_x if lastFrame_x>lastFrame_y else lastFrame_y
    if lastFrame!=df.iloc[-1][0]: df = df.iloc[:lastFrame]
        
    return df, int(lastFrame)


datasets = ['train', 'dev', 'test']
for i in range(1,3):
    base_dir = os.path.join('dataset','ElderReact_Data',f'ElderReact_{datasets[i]}','')
    for video in os.listdir(base_dir):
        videoName = video[:-4]
        openface_df = pd.read_csv(os.path.join('openFace', datasets[i],'processed',f'{videoName}_openface.csv'))
        mediapipe_df = pd.read_csv(os.path.join('mediaPipe', datasets[i],'processed', f'{videoName}_mediapipe.csv'))
        
        openface_df, openface_firstFrame = remove_first_frames(openface_df)
        mediapipe_df, mediapipe_firstFrame = remove_first_frames(mediapipe_df)
        
        if openface_firstFrame!=0:
            mediapipe_df = mediapipe_df.iloc[openface_firstFrame:, :]
        if mediapipe_firstFrame!=0:
            openface_df = openface_df.iloc[mediapipe_firstFrame:, :]
        
        openface_df, openface_lastFrame = remove_last_frames(openface_df)
        mediapipe_df, mediapipe_lastFrame = remove_last_frames(mediapipe_df)
        
        if openface_lastFrame!=openface_df.iloc[-1][0]:
            mediapipe_df = mediapipe_df.iloc[:openface_lastFrame, :]
        if mediapipe_lastFrame!=mediapipe_df.iloc[-1][0]:
            openface_df = openface_df.iloc[:mediapipe_lastFrame, :]
        
        openface_df.to_csv(os.path.join('openFace',datasets[i],'processed_cleansed',f'{videoName}_openface.csv'), index=False)
        mediapipe_df.to_csv(os.path.join('mediaPipe',datasets[i],'processed_cleansed',f'{videoName}_mediapipe.csv'), index=False)
