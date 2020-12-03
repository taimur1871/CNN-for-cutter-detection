""" module to import data for training and testing"""

import tensorflow as tf
import numpy as np

'''train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        directory = '/home/taimur/Pictures/Cutter_or_Not/Train',
        target_size=(300, 300),
        batch_size=32,
        class_mode='binary')'''

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        directory = '/home/taimur/Pictures/Cutter_or_Not/Test',
        target_size=(300, 300),
        batch_size=32,
        class_mode='binary')


for x in test_generator:
    print(x)
