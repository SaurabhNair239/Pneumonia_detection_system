import pandas as pd
import streamlit as slt
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
import pickle


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
        image = load_img(img,target_size=(224,224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        my_image = preprocess_input(image)
        filename = 'cnn_model.pickle'
        model = pickle.load(open(filename, 'rb'))
        result = model.predict(my_image)
        if result == 1:
            slt.write("The models detects Pneumonia")
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

