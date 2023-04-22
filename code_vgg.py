from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop,SGD,Adam
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np
import os
import sys
from PIL import Image
sys.modules['Image'] = Image 
import cv2
tf.compat.v1.set_random_seed(2019)

rows, cols = 110,110
input_shape = (rows,cols,1)

train = ImageDataGenerator(rescale= 1/255)
validation = ImageDataGenerator(rescale= 1/255)
test = ImageDataGenerator(rescale= 1/255)
train_dataset = train.flow_from_directory(r"C:\Users\retr0\Desktop\CCR\new_dat\Train",target_size=(110,110), batch_size=30, class_mode = 'categorical', color_mode = 'grayscale')
validation_dataset = validation.flow_from_directory(r"C:\Users\retr0\Desktop\CCR\new_dat\Validation",target_size=(110,110), batch_size=30, class_mode = 'categorical', color_mode = 'grayscale')
test_dataset = test.flow_from_directory(r"C:\Users\retr0\Desktop\CCR\new_dat\Test",target_size=(110,110), batch_size=30, class_mode = 'categorical', color_mode = 'grayscale')

#VGGNet
model2 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64,(3,3),activation='relu', input_shape=input_shape),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
    tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(256,(3,3),activation='relu'),
    tf.keras.layers.Conv2D(256,(3,3),activation='relu'),
    tf.keras.layers.Conv2D(256,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(512,(3,3),activation='relu'),
    tf.keras.layers.Conv2D(512,(3,3),activation='relu'),
    tf.keras.layers.Conv2D(512,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(500, activation='relu'),
    tf.keras.layers.Dense(120, activation='relu'),
    tf.keras.layers.Dense(84,activation='relu'),
    tf.keras.layers.Dense(3,activation='softmax')
    ])


model2.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['acc'])
print("")
model2.summary()

bs=80
epochs=10

model2_fit = model2.fit(train_dataset,
                       validation_data=test_dataset,
                       steps_per_epoch=8500 // bs,
                       epochs=epochs,
                       validation_steps=600 // bs)

validation_dataset.reset()

model2.evaluate(validation_dataset)


