import streamlit as st
from fastai.vision.all import *
from fastai.vision.all import PILImage
from fastai.learner import load_learner
import pathlib
from pathlib import Path
import plotly
import plotly.express as px
import platform

plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.WindowsPath


import os

# Function to convert WindowsPath to PosixPath if needed
def convert_path(path):
    if isinstance(path, Path):
        path = str(path)
    return path.replace('\\', '/') if os.name != 'nt' else path

model_path = 'transport_model.pkl'
model_path = convert_path(model_path)

if not os.path.exists(model_path):
    print(f"Error: Model file {model_path} does not exist.")
else:
    print(f"Model file {model_path} found.")

import torch

try:
    model = torch.load(model_path, map_location='cpu')
    print("Model loaded successfully with torch.")
except Exception as e:
    print(f"Error loading model with torch: {e}")


#title
st.title("Transportni klassifikatsiya qiluvchi model")


# rasmni joylash
file = st.file_uploader("Rasm yuklash", type=['png', 'jpg', 'gif', 'svg'])
if file:
    st.image(file)
    # PIL convert
    img = PILImage.create(file)
    
    # modelni yuklash
    model = load_learner('transport_model.pkl')
    
    # bashorat qilish prediction
    pred, pred_id, probs = model.predict(img)
    st.success(f"Bashorat: {pred}")
    st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")
    
    # plotting
    
    fig = px.bar(x= probs*100, y = model.dls.vocab)
    st.plotly_chart(fig)