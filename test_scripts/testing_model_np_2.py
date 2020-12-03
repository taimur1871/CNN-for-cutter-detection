# python 3

import tensorflow as tf
from tensorflow.keras import models
import numpy as np
import matplotlib.image as mpimg
import os
from PIL import Image

folder = input('Enter file address\n')
os.chdir(folder)

file_list = os.listdir(folder)
new_model = tf.keras.models.load_model('model_path')

for i, file in enumerate(file_list):
    file_temp = Image.open(file)
    file_1 = file_temp.resize((200,200))
    file_name = np.asanyarray(file_1)
    file_name = file_name/255.
    test_file = np.expand_dims(file_name, 0)
    temp = new_model.predict(test_file)
    if temp > 0.7:
        os.rename(file, 'not cutter' + str(i))
    else:
        os.rename(file, 'cutter'+ str(i))
