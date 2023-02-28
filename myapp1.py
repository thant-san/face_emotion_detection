import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image,ImageOps
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import model_from_json

#load_model
json_file = open("model.json", "r")
loaded_json_model = json_file.read()
json_file.close()
model = model_from_json(loaded_json_model)
model.load_weights("model_weights.h5")

file_upload=st.file_uploader("insert")
if file_upload is None:
  st.write("input image")
else:
  img=Image.open(file_upload)

  size=(48,48)
  img=ImageOps.fit(img, size, Image.ANTIALIAS)
  st.image(img)
  img=ImageOps.grayscale(img)
  #st.image(img)
  img=np.asarray(img)
  st.write(img.shape)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  emotions = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')
  max_index=np.argmax(model.predict(img.reshape((1,48,48,1))), axis=-1)[0]
  predicted_emotion = emotions[max_index] 
  st.write(f'predicted emotion is {predicted_emotion}')
import streamlit as st
import cv2
import numpy as np

img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    # To read image file buffer with OpenCV:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # Check the type of cv2_img:
    # Should output: <class 'numpy.ndarray'>
    st.write(type(cv2_img))

    # Check the shape of cv2_img:
    # Should output shape: (height, width, channels)
    st.write(cv2_img.shape)
    img=ImageOps.grayscale(cv2_img)
  #st.image(img)
    img=np.asarray(img)
    st.write(img.shape)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    emotions = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')
    max_index=np.argmax(model.predict(img.reshape((1,48,48,1))), axis=-1)[0]
    predicted_emotion = emotions[max_index] 
    st.write(f'predicted emotion is {predicted_emotion}')
