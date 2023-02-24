import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image,ImageOps
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
model=load_model('Fer_Model.h5')
file_upload=st.file_uploader("insert")
img=Image.open(file_upload)
size=(277,277)
img=ImageOps.fit(img,size,Image.ANTIALIAS)
st.image(img)
img=cv2.imread(img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
emotions = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')
string=np.argmax(model.predict(img.reshape((1,48,48,1))), axis=-1)[0]
st.write('string')
