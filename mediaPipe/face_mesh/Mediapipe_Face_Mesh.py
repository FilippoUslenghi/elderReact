# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# Usage example of MediaPipe Face Detection Solution API in Python (see also http://solutions.mediapipe.dev/face_detection).

# %%
# get_ipython().system('pip install mediapipe')

# %% [markdown]
# Upload any image that contains face(s) to the Colab. We take two example images from the web: https://unsplash.com/photos/JyVcAIUAcPM and https://unsplash.com/photos/auTAb39ImXg

# %%
import os
import cv2
import plotly.express as px
import matplotlib.pyplot as plt
# from google.colab.patches import cv2_imshow
import mediapipe as mp
from IPython.display import clear_output

# %% [markdown]
# # Estrazione landmark su 'frontal_face.png'

# %%
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# For static images:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=0)
with mp_face_mesh.FaceMesh(
  static_image_mode=True,
  max_num_faces=1,
  min_detection_confidence=0.5) as face_mesh:

  image = cv2.imread('frontal_face.png')
  # Convert the BGR image to RGB before processing.
  results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

  # Print and draw face mesh landmarks on the image.
  if not results.multi_face_landmarks: print('Error')
  annotated_image = image.copy()
  for face_landmarks in results.multi_face_landmarks:
    # print('face_landmarks:', face_landmarks)
    mp_drawing.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACE_CONNECTIONS,
        landmark_drawing_spec=drawing_spec,
        connection_drawing_spec=drawing_spec)
  annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
  plt.figure(figsize=(8,8))
  plt.imshow(annotated_image)
  plt.show()


# %%
landmarks = face_landmarks.landmark
for landmarks in results.multi_face_landmarks:
  idx=[i for i in range(468)]
  x=[landmark.x for landmark in landmarks.landmark]
  y=[landmark.y for landmark in landmarks.landmark]

x = [int(i*451) for i in x]
y = [int(i*259) for i in y]

points = [(i,x,y) for i,x,y in zip(idx,x,y)]

# %% [markdown]
# Funzione per estrare un landmark date le sue coordinate

# %%
def extract_landmark(x, y):
  print([point for point in points if x == point[1] and y == point[2]])

extract_landmark(228,165)


# %%
img = cv2.imread('frontal_face_annotated.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
for x in range(451):
    for y in range(259):
        if not (img[y,x,0]==0 and img[y,x,1]==255 and img[y,x,2]==0):
            img[y,x,0]=0
            img[y,x,1]=0
            img[y,x,2]=0
for point in points:

    img[point[2],point[1],0] = 255
    img[point[2],point[1],1] = 0
    img[point[2],point[1],2] = 0
    if point[0] >= 185:
      print(point)
      fig = px.imshow(img)
      fig.show()
      next = input('Next landmark...')
      clear_output(wait=True)
    img[point[2],point[1],0] = 255
    img[point[2],point[1],1] = 255
    img[point[2],point[1],2] = 255

# %%
