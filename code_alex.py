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

rows, cols = 360,360
input_shape = (rows,cols,1)

train = ImageDataGenerator(rescale= 1/255)
validation = ImageDataGenerator(rescale= 1/255)
test = ImageDataGenerator(rescale= 1/255)
train_dataset = train.flow_from_directory(r"C:\Users\retr0\Desktop\CCR\new_dat\Train",target_size=(50,50), batch_size=30, class_mode = 'categorical', color_mode = 'grayscale')
validation_dataset = validation.flow_from_directory(r"C:\Users\retr0\Desktop\CCR\new_dat\Validation",target_size=(50,50), batch_size=30, class_mode = 'categorical', color_mode = 'grayscale')
test_dataset = test.flow_from_directory(r"C:\Users\retr0\Desktop\CCR\new_dat\Test",target_size=(50,50), batch_size=30, class_mode = 'categorical', color_mode = 'grayscale')

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=96,input_shape=input_shape, kernel_size=(11,11), strides=(4,4),padding='valid', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(256,kernel_size=(11,11), strides=(1,1), padding='valid', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(384,(3,3), strides=(1,1), padding='valid', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(384,(3,3), strides=(1,1), padding='valid', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(256,(3,3), strides=(1,1), padding='valid', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(200, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(3, activation='softmax')
    ])

model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['acc'])
model.summary()

bs=80
epochs=10

model_fit = model.fit(train_dataset,
                       validation_data=test_dataset,
                       steps_per_epoch=8500 // bs,
                       epochs=epochs,
                       validation_steps=600 // bs)

validation_dataset.reset()

model.evaluate(validation_dataset)


