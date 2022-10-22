import os
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D, Flatten
import tensorflow as tf
import pickle
#from tensorflow.keras.application.vgg16 import VGG16


test_folder = "test"
train_folder = "train"
valuation_folder = "val"

train_normal = os.path.join(train_folder, 'NORMAL')
train_pneumonia = os.path.join(train_folder, 'PNEUMONIA')

test_normal = os.path.join(test_folder, 'NORMAL')
test_pneumonia = os.path.join(test_folder, 'PNEUMONIA')


val_normal = os.path.join(valuation_folder, 'NORMAL')
val_pneumonia = os.path.join(valuation_folder, 'PNEUMONIA')


train_len_normal = len(os.listdir(train_normal))
train_len_pneumonia = len(os.listdir(train_pneumonia))
print(train_len_normal,train_len_pneumonia)

test_len_normal = len(os.listdir(test_normal))
test_len_pneumonia = len(os.listdir(test_pneumonia))
print(test_len_normal,test_len_pneumonia)

first_train_normal_img = os.path.join(train_normal,os.listdir(train_normal)[0])
image = plt.imread(first_train_normal_img)
#plt.imshow(image)


train_data = glob(train_folder+"/PNEUMONIA/*.jpeg") + glob(train_folder+"/NORMAL/*.jpeg")
test_data = glob(test_folder+"/PNEUMONIA/*.jpeg") + glob(test_folder+"/NORMAL/*.jpeg")
val_data = glob(valuation_folder+"/PNEUMONIA/*.jpeg") + glob(valuation_folder+"/NORMAL/*.jpeg")

rescaling = ImageDataGenerator(rescale=1/255)

training_set = rescaling.flow_from_directory(train_folder,target_size=(128,128), batch_size=10,class_mode="binary")
test_set = rescaling.flow_from_directory(test_folder,target_size=(128,128), batch_size=10,class_mode="binary")
validation_set = rescaling.flow_from_directory(valuation_folder,target_size=(128,128), batch_size=10,class_mode="binary")
"""
model = models.Sequential()
model.add(Conv2D(128, (3, 3), activation="relu", input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())

# activation
model.add(Dense(activation = 'relu', units = 128))
model.add(Dense(activation = 'relu', units = 32))
model.add(Dense(activation = 'sigmoid', units = 1))
model.add(layers.Flatten())

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0.001,
    patience=20,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
)



model_history = model.fit(training_set, batch_size=200,epochs = 15, validation_data = validation_set, validation_split=0.33,callbacks=early_stopping)


plt.plot(model_history.history["accuracy"])
plt.plot(model_history.history["val_accuracy"])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Model accuracy")
plt.legend(["train","test"],loc = "upper left")
plt.show()

plt.plot(model_history.history["loss"])
plt.plot(model_history.history["val_loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Model loss")
plt.legend(["train","test"],loc = "upper left")
plt.show()

model_history_data = pd.concat([model_history["loss"],model_history["val_loss"],model_history["accuracy"],model_history["val_accuracy"]],axis=1)

filename = 'cnn_model.pickle'
pickle.dump(model, open(filename, 'wb'))
"""

base_model = tf.keras.applications.vgg16.VGG16(input_shape = (128, 128, 3), include_top = False, weights = 'imagenet')

for layer in base_model.layers:
    layer.trainable = False

x = layers.Flatten()(base_model.output)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(32, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.models.Model(base_model.input, x)

model.compile(optimizer = "adam", loss = 'binary_crossentropy',metrics = ['acc'])
vgghist = model.fit(training_set, validation_data = validation_set, steps_per_epoch = 100, epochs = 10)

plt.plot(vgghist.history["acc"])
plt.plot(vgghist.history["val_acc"])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Model accuracy")
plt.legend(["train","test"],loc = "upper left")
plt.show()

plt.plot(vgghist.history["loss"])
plt.plot(vgghist.history["val_loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Model loss")
plt.legend(["train","test"],loc = "upper left")
plt.show()

file_name = "vgg16_model.h5"
model.save(file_name)
model_history_data = pd.DataFrame({"loss":vgghist.history["loss"],
                                   "val_loss":vgghist.history["val_loss"],
                                   "acc":vgghist.history["acc"],
                                   "val_acc":vgghist.history["val_acc"]})
model_history_data.to_csv("model_history.csv")
