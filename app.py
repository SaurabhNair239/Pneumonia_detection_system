import pandas as pd
import streamlit as slt
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
import pickle
from prediction_file import binary_classification

slt.title("Pneumonia Classification Deep learning model")

slt.header("Teaching systems to predict, If X_ray contains Pneumonia or not")
image_types = ["png", "jpeg","jpg"]
file = slt.file_uploader("Please Upload the X_ray of chest",type=image_types)
slt.sidebar.subheader("Accuracy and loss of a model")
slt.sidebar.line_chart(pd.read_csv("model_history.csv")["acc"])
slt.sidebar.line_chart(pd.read_csv("model_history.csv")["loss"])

if file is not None:
    img = Image.open(file)
    slt.write("Your result will be soon displayed. Predicting..")
    slt.spinner()
    with slt.spinner(text="Work in Progress..."):
        result = binary_classification(img)
        if result == 1:
            slt.write("The models detects Pneumonia. Please consult the doctor soon.")
            slt.write("")
            slt.write("")
            slt.write("")
            slt.write("")
            slt.write("")
            slt.write("Still it is always recommended to consult a doctor ")
        else:
            slt.write("The models was unable to detect Pneumonia")
            slt.write("")
            slt.write("")
            slt.write("")
            slt.write("")
            slt.write("")
            slt.write("It is always recommended to consult a doctor ")

