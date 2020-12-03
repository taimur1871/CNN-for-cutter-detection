import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import tqdm

folder = input('Enter Folder Path\n')
model_input= tf.keras.models.load_model('/home/taimur/Pictures/New Bit Classification/Model/3c-4l_densenet-more data.h5')

uploaded = os.listdir(folder)
k=0
cutter = 0
blade = 0
top = 0

for i in tqdm.trange(len(uploaded)):
  # predicting images
  path = folder+'/'+uploaded[i]
  img = image.load_img(path, target_size=(300, 300))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)

  images = np.vstack([x])
  classes = model_input.predict(images)
  print(uploaded[i])

  #for classification
  if (classes[0][0] == max(classes[0])):
    print('blade')
    blade += 1
  elif (classes[0][1] == max(classes[0])):
    print('cutter')
    cutter += 1
  elif (classes[0][2] == max(classes[0])):
    print('top')
    top += 1
  '''#for renaming
  if (classes[0][0] == max(classes[0])):
    os.rename(path, folder+'/'+'bl'+str(k)+'.jpg')
    blade += 1
  elif (classes[0][1] == max(classes[0])):
    os.rename(path, folder+'/'+'cut'+str(k)+'.jpg')
    cutter += 1
  if (classes[0][2] == max(classes[0])):
    os.rename(path, folder+'/'+'tp'+str(k)+'.jpg')
    top += 1'''
  k+=1

print('cutter: ', cutter)
print('blade: ', blade)
print('top: ', top)
