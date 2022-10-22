from tensorflow.keras.preprocessing.image import img_to_array,load_img
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_vgg
from PIL import Image
import numpy as np
import tensorflow as tf


def binary_classification(img):
    input_shape = (128,128)
    vgg_model = tf.keras.models.load_model("vgg16_model.h5")
    image_rgb = img.convert('RGB').resize(input_shape, Image.ANTIALIAS)
    image_rgb.load()
    image_rgb_array = np.array(image_rgb)
    expand_image_array = np.expand_dims(image_rgb_array, axis=0)
    preprocessed_expanded_image_array = preprocess_input_vgg(expand_image_array)
    y_pred = vgg_model.predict(preprocessed_expanded_image_array)
    output_label = [1 if x > 0.5 else 0 for x in y_pred]
    return output_label[0]