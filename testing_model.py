# python 3

import tensorflow as tf
from tensorflow.keras import models
import numpy as np
import os

import folder_select

target = folder_select.root.directory

predict_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

predict_generator = predict_datagen.flow_from_directory(
        directory = target,
        target_size=(200, 200))

new_model = tf.keras.models.load_model('/home/taimur/Pictures/Cutter_or_Not/Model/model2')

# Check architecture
new_model.summary()

results = new_model.predict(predict_generator)

#file1 = open(new_file.txt, 'w')
#file1.write(str(results))
