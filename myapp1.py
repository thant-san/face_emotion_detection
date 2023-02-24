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

  size=(227,227)
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
