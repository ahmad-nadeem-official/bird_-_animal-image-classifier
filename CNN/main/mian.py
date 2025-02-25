from ast import Import
import pandas as pd
import numpy as np
from google.colab.patches import cv2_imshow
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tf import keras
import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import cv2
import os
from zipfile import ZipFile

ds = r'/content/data(CNN).zip'
with ZipFile(ds, 'r') as zip:
  zip.extractall()
  print('Done')


eagle = os.listdir(r'/content/data(CNN)/bird bald_eagle')
hare = os.listdir(r'/content/data(CNN)/hare')

eagle[0:5], eagle[-5:]
# (['SR6WSBGCMN2H.jpg',
#   'O7C12ZGKJCBD.jpg',
#   'VGA0U5YO5VCD.jpg',
#   'INPLQ40Y6XND.jpg',
#   'F6CGZ3TTJNXH.jpg'],
#  ['L1M5RZY639ZC.jpg',
#   '2W0ZP3GNIBOM.jpg',
#   'FR0MVZ1TT2J9.jpg',
#   '52J88WMBJQAN.jpg',
#   '3CPVITW20T35.jpg'])

hare[0:5], hare[-5:]
# (['P3S6U2EKQEQ1.jpg',
#   '6NF3USJR5FU2.jpg',
#   '2I0S6EWYEDD1.jpg',
#   '2YOBTTW0OQKH.jpg',
#   'DWXQ204PP1SA.jpg'],
#  ['FGAB5NLV38WB.jpg',
#   'MWVFV3DCSIRD.jpg',
#   '9JUK72TIGP42.jpg',
#   'XZWUBHTZLW0Y.jpg',
#   'QJQ0BW2U1MU6.jpg'])

len(eagle), len(hare), sum([len(eagle), len(hare)])
# (1460, 1356, 2816)

eagle_img = [1]*len(eagle)
hare_img = [0]*len(hare)
label = eagle_img + hare_img
label[0:5], label[-5:]
len(label)

img = mpimg.imread(r'/content/data(CNN)/bird bald_eagle/SR6WSBGCMN2H.jpg')
plt.imshow(img)

img2 = mpimg.imread(r'/content/data(CNN)/hare/P3S6U2EKQEQ1.jpg')
plt.imshow(img2)

data = []

rabit = r'/content/data(CNN)/hare'
for i in os.listdir(rabit):
    file_path = os.path.join(rabit, i)
    if file_path.lower().endswith(('png', 'jpg', 'jpeg')):
        image = Image.open(file_path)
        image = image.resize((128, 128))
        image = image.convert('RGB')
        image = np.array(image)
        data.append(image)

bird = r'/content/data(CNN)/bird bald_eagle'
for y in os.listdir(bird):
    file_path1 = os.path.join(bird, y)
    if file_path1.lower().endswith(('png', 'jpg', 'jpeg')):
        image = Image.open(file_path1)
        image = image.resize((128, 128))
        image = image.convert('RGB')
        image = np.array(image)
        data.append(image)

len(data)

x = np.array(data)
y = np.array(label)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

x_train.shape, x_test.shape, y_train.shape, y_test.shape
# ((2252, 128, 128, 3), (564, 128, 128, 3), (2252,), (564,))

x_train_scale = x_train/255
x_test_scale = x_test/255

x_train_scale[0] ,x_test_scale.shape[0]


no_of_class = 2
model = keras.Sequential()

model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(keras.layers.MaxPooling2D((2, 2)))

model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(no_of_class, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train_scale, y_train, epochs=10, validation_data=(x_test_scale, y_test), validation_split=0.1)

loss, accuracy = model.evaluate(x_test_scale, y_test)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

# Loss: 0.4206562042236328
# Accuracy: 0.8652482032775879

h = history

plt.plot(h.history['loss'], label='train loss')
plt.plot(h.history['val_loss'], label='val loss')
plt.legend()
plt.show()

plt.plot(h.history['accuracy'], label='train acc')
plt.plot(h.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()

pred_img = r'/content/data(CNN)/hare/P3S6U2EKQEQ1.jpg'
immg = cv2.imread(pred_img)
cv2.imshow(immg)
immg_resize = cv2.resize(immg, (128, 128))
immg_scale = immg_resize/255
immg_reshape = np.reshape(immg_scale, [1, 128, 128, 3])
pred = model.predict(immg_reshape)
print(pred)
pred_label = np.argmax(pred)
print(pred_label)
if pred_label == 0:
  print('hare')
else:
  print('bird')

pred_img = r'/content/data(CNN)/bird bald_eagle/SR6WSBGCMN2H.jpg'
immg = cv2.imread(pred_img)
cv2_imshow(immg)
immg_resize = cv2.resize(immg, (128, 128))
immg_scale = immg_resize/255
immg_reshape = np.reshape(immg_scale, [1, 128, 128, 3])
pred = model.predict(immg_reshape)
print(pred)
pred_label = np.argmax(pred)
print(pred_label)
if pred_label == 0:
  print('hare')
else:
  print('bird')  



