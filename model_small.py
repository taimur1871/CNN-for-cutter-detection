'''tensorflow model for training cutter recognition'''

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt

#training data
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        directory = '/home/taimur/Pictures/Cutter_or_Not/Train',
        target_size=(200, 200),
        class_mode='binary')

#validation data
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(
        directory = '/home/taimur/Pictures/Cutter_or_Not/Validation',
        target_size=(200, 200),
        class_mode='binary')

#model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=RMSprop(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()
history = model.fit(train_generator, steps_per_epoch = 20, epochs=100, 
                    validation_data=validation_generator, validation_steps=5)

model.save('/home/taimur/Pictures/Cutter_or_Not/Model/model2')

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([1, 0])
plt.legend(loc='upper right')
plt.show()

#test_loss, test_acc = model.evaluate(test_generator, verbose=2)
