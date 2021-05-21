# %%
import os
import cv2
import mediapipe as mp

# %%
base_dir ='C:\\Users\\us98\\PycharmProjects\\elderReactProject\\dataset\\ElderReact_Data\\ElderReact_train\\'
out_dir = 'C:\\Users\\us98\\PycharmProjects\\elderReactProject\\mediaPipe_videos\\'
videos = os.listdir(base_dir)

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

for i, video in enumerate(videos):
    
    if os.path.isdir(video): continue
    videoName = video[:-4]
    cap = cv2.VideoCapture(base_dir + video)
    img_array = []

    if videoName + '_mediapipe.avi' in os.listdir(out_dir): continue

    unknown_size = True
    while cap.isOpened():

        success, image = cap.read()
        if not success:
            print("Error")
            # If loading a video, use 'break' instead of 'continue'.
            break

        with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

            # Convert the BGR image to RGB.
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACE_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)

        # Create a video with the landmark
        height, width, layers = image.shape
        size = (width,height)
        
        if unknown_size:
            out = cv2.VideoWriter(out_dir + videoName + '_mediapipe.avi', cv2.VideoWriter_fourcc(*'DIVX'), 24, size)
            unknown_size = False
            
        out.write(image)
        
    print(i, '\t', videoName)


    cap.release()

# %%
