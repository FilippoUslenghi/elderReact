import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths
from sklearn.metrics import mean_squared_error

FACE_MESH = pd.read_csv(os.path.join('faceMesh','face_mesh.csv'))
def landmarks_mapper(of_df, mp_df):
    
    OPENFACE_LANDMARKS = [i for i in range(17, 68)]
    
    final_df = of_df['frame'] # initialize the final dataframe
    for landmark in OPENFACE_LANDMARKS:

        mask = (FACE_MESH['openFace_landmark'] == landmark) # group the mediapipe landmarks by the openface landmark

        tmp1_df = of_df[[f'x_{landmark}', f'y_{landmark}']] # select the columns from the openface dataframe
        tmp1_df.columns = [f'openface_x_{landmark}', f'openface_y_{landmark}'] # rename the columns
        
        mediaPipe_landmarks_X = ['x_'+str(ID) for ID in FACE_MESH[mask].id] # collect the X coordinate
        mediaPipe_landmarks_Y = ['y_'+str(ID) for ID in FACE_MESH[mask].id] # collect the Y coordinate
        
        tmp2_df = pd.DataFrame({f'mediapipe_x_{landmark}':mp_df[mediaPipe_landmarks_X].mean(axis=1),
                                f'mediapipe_y_{landmark}':mp_df[mediaPipe_landmarks_Y].mean(axis=1)})
        
        final_df = pd.concat([final_df, tmp1_df, tmp2_df], axis=1)
        
    return final_df

def mse(df: pd.DataFrame):
    
    lndmk_mse = []
    for i in range(17, 68):
        
        x_mse = mean_squared_error(df[f'mediapipe_x_{i}'].to_numpy(), df[f'openface_x_{i}'].to_numpy())
        y_mse = mean_squared_error(df[f'mediapipe_y_{i}'].to_numpy(), df[f'openface_y_{i}'].to_numpy())
        lndmk_mse.append((x_mse+y_mse)/2)
    
    video_mse = np.mean(lndmk_mse)
    
    return video_mse

def clean_and_interpolate(openface_df, mediapipe_df, threshold):
    
    """
    If the video is worth saving, returns the dataframe cleansed and interpolated
    else returns None
    """
    ACTION_UNITS = ['AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r']
    
    tmp_df=openface_df # create a copy of the openface dataframe
    mapped_landmarks_df = landmarks_mapper(openface_df, mediapipe_df)
    window_size = (lambda seconds: round(24*seconds))(0.2)
    windows = mapped_landmarks_df.rolling(window=window_size, center=True) # centralized sliding window
    windowized_mse = np.array([mse(window) for window in windows])
    
    first_frame = mapped_landmarks_df.frame.iloc[0]
    # removing head and tail of the dataframe if necessary
    start_frame, end_frame = None, None
    if windowized_mse[0]>threshold:
        for index, mse_value in enumerate(windowized_mse):
            if mse_value < threshold:
                start_frame = index+first_frame
                break

    if windowized_mse[-1]>threshold:
        for index in range(len(windowized_mse)-1, -1, -1):
            if windowized_mse[index] < threshold:
                end_frame = index+first_frame
                break

    if start_frame:
        openface_df.drop([frame for frame in range(start_frame-first_frame+1)], inplace=True)
        mediapipe_df.drop([frame for frame in range(start_frame-first_frame+1)], inplace=True)
    if end_frame:
        openface_df.drop([frame for frame in range(end_frame-first_frame, openface_df.frame.iloc[-1]-first_frame+1)], inplace=True)
        mediapipe_df.drop([frame for frame in range(end_frame-first_frame, int(mediapipe_df.frame.iloc[-1])-first_frame+1)], inplace=True)
        
    if start_frame or end_frame:
        # compute again the windowized MSE with the new dataframe
        mapped_landmarks_df = landmarks_mapper(openface_df, mediapipe_df)
        window_size = (lambda seconds: round(24*seconds))(0.2)
        windows = mapped_landmarks_df.rolling(window=window_size, center=True) # centralized sliding window
        windowized_mse = [mse(window) for window in windows]
    
    # finding peaks
    peaks, _ = find_peaks(windowized_mse, height=threshold)
    widths = peak_widths(windowized_mse, peaks, rel_height=0.90)
    for frames in widths[2:]:
        frames += first_frame # shift the peaks by the starting frame number

    starting_points = widths[2]
    ending_points = widths[3]
    
    # interpolation
    for peak_points in zip(starting_points, ending_points):

        start = round(peak_points[0])
        end = round(peak_points[1])
        
        tmp_df.loc[start:end-2, f'x_27'] = np.nan # in this way i can then count the number of rows (and so frames) that have to be interpolated
        for i in range(68):
            openface_df.loc[start:end-2, f'x_{i}'], openface_df.loc[start:end-2, f'y_{i}'] = np.nan, np.nan
        for i in range(468):
            mediapipe_df.loc[start:end-2, f'x_{i}'], mediapipe_df.loc[start:end-2, f'y_{i}'] = np.nan, np.nan
            
        
        for action_unit in ACTION_UNITS:
            openface_df.loc[start:end-2, action_unit] = np.nan
            
    # check if the video is worth saving
    n_frames_interpolated = np.count_nonzero(tmp_df.x_27.isnull())  # count the number of rows (and so frames) that have to be interpolated
    if n_frames_interpolated/tmp_df.frame.size>0.6: return None, None, 0  # if they are over the 60% of the video, reject the video

    openface_df.interpolate(method='linear', axis=0, inplace=True)
    mediapipe_df.interpolate(method='linear', axis=0, inplace=True)
        
    return openface_df, mediapipe_df, n_frames_interpolated



THRESHOLD = 75
columns = ['frame','AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r','AU01_c','AU02_c','AU04_c','AU05_c','AU06_c','AU07_c','AU09_c','AU10_c','AU12_c','AU14_c','AU15_c','AU17_c','AU20_c','AU23_c','AU25_c','AU26_c','AU28_c','AU45_c']
datasets = ['train','dev','test']
for dataset in datasets:
    base_dir = os.path.join('dataset','ElderReact_Data',f'ElderReact_{dataset}')
    
    for video in os.listdir(base_dir):
        
        video_name = video[:-4]       
        openface_df = pd.read_csv(os.path.join('openFace', dataset, 'processed', f'{video_name}_openface.csv'))
        mediapipe_df = pd.read_csv(os.path.join('mediaPipe', dataset, 'processed', f'{video_name}_mediapipe.csv'))
        
        openface_df, mediapipe_df, n_frames_interpolated = clean_and_interpolate(openface_df, mediapipe_df, THRESHOLD)
        if openface_df is not None:
            openface_df.to_csv(os.path.join('openFace', dataset, 'interpolated', f'{video_name}_openface.csv'), index=False)
        if mediapipe_df is not None:
            mediapipe_df.to_csv(os.path.join('mediaPipe', dataset, 'interpolated', f'{video_name}_mediapipe.csv'), index=False)
        else: print(video_name)
