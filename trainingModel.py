import tensorflow as tf
from focal_loss import BinaryFocalLoss

from tensorflow import keras
from tensorflow.keras import utils
from tensorflow.keras import layers
from tensorflow.keras import datasets
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
from tensorflow.keras.models import Sequential

import numpy as np
import math
import matplotlib.pyplot as plt
from glob import glob
import cv2

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

import warnings  # 경고문 뜨지않게 하는 코드

warnings.filterwarnings('ignore')

image_datas = glob('C:/Users/이서정/PycharmProjects/OcrProject/phd08_png_results1/*/*.jpg')
#image_datas = glob('.\\phd08_png_results1\\*\\*')
#image_datas=glob('D:\POLARIS\png2jpg\*.jpg')
class_datas = glob('C:\\Users\\이서정\\PycharmProjects\\OcrProject\\phd08_png_results1\\*')
txtN = len(glob('C:\\Users\\이서정\\PycharmProjects\\OcrProject\\phd08_png_results1\\*'))

class_name = []

for i in range(txtN):
    class_name.append(class_datas[i][59])
dic = {}
for j in range(len(class_name)):
    a = str(class_name[j])
    dic[a] = j


X, Y = [], []

for i in image_datas:
    image = cv2.imread(i)
    # image = open(i)
    image = np.array(image)
    X.append(image)
    label = i.split('\\')[6]
    label = dic[label]
    Y.append(label)

X = np.array(X)
Y = np.array(Y)

train_images, etc_images, train_labels, etc_labels = train_test_split(X, Y, test_size=0.2, random_state=42)
val_images, test_images, val_labels, test_labels = train_test_split(etc_images, etc_labels, test_size=0.5,
                                                                    random_state=42)

print(len(train_images), len(val_images), len(test_images))

train_labels = train_labels[..., tf.newaxis]
val_labels = val_labels[..., tf.newaxis]
test_labels = test_labels[..., tf.newaxis]

train_images.shape, train_labels.shape, val_images.shape, val_labels.shape, test_images.shape, test_labels.shape

len_train = len(train_labels)
len_val = len(val_labels)
len_test = len(test_labels)

# unique, counts = np.unique(np.reshape(train_labels, (len_train,)), axis=-1, return_counts=True) #여기
# dict(zip(unique, counts))

# print(unique, counts)

N_TRAIN = train_images.shape[0]
N_VAL = val_images.shape[0]
N_TEST = test_images.shape[0]

train_images = train_images.astype(np.float32) / 255
val_images = val_images.astype(np.float32) / 255
test_images = test_images.astype(np.float32) / 255

train_labels = keras.utils.to_categorical(train_labels)
val_labels = keras.utils.to_categorical(val_labels)
test_labels = keras.utils.to_categorical(test_labels)

print(train_images.shape, train_labels.shape)
print(val_images.shape, val_labels.shape)
print(test_images.shape, test_labels.shape)


#####################################################################################

N_BATCH = 32

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(buffer_size=len_train).batch(
    N_BATCH)
val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).shuffle(buffer_size=len_val).batch(N_BATCH)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).shuffle(buffer_size=len_test).batch(
    N_BATCH)

# 모델 선언
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=train_images.shape[1:], padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 전결합층
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(3))
model.add(Activation('softmax'))

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)
mc = tf.keras.callbacks.ModelCheckpoint('motorcycle_best_model.h5', monitor='val_loss', verbose=1, mode='min',
                                        save_best_only=True)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# 파라미터
N_EPOCHS = 1

# 모델 훈련
history = model.fit(train_dataset, epochs=N_EPOCHS, validation_data=val_dataset, batch_size=N_BATCH, callbacks=[es, mc])

# 모델 평가
loss, acc = model.evaluate(test_dataset)
print('loss :', loss, '\n Acc :', acc)

predict = model.predict(test_dataset)
for i in range(len(test_dataset)):
    print(test_labels[i])
    print(np.argmax(predict[i]))
print(len(test_dataset))

########## 그림으로 loss, acc 보기 ################
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(23) # Epoch이 돌아가는 수에 맞춰서 변경해줘야함.

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
