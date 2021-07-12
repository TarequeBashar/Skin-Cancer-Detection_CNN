import numpy as np
import pandas as pd
import random
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, BatchNormalization
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, log_loss, accuracy_score
import matplotlib.pyplot as plt

data = pd.read_csv('./data/hmnist_28_28_RGB.csv')
n=len(data)
N=[]
for i in range(n):
    N+=[i]
random.shuffle(N)
Data = data.iloc[:,0:-1]
Label = data.iloc[:,-1]

X=Data.iloc[N[0:(n//10)*8],:]
y0=Label[N[0:(n//10)*8]]

test=Data.iloc[N[(n//10)*8:],:]
test_label=Label[N[(n//10)*8:]]
binencoder = LabelBinarizer()
y = binencoder.fit_transform(y0)

X_images = X.values.reshape(-1,28,28,3)
test_images = test.values.reshape(-1,28,28,3)

print(X_images.shape)
print(test_images.shape)
(8008, 28, 28, 3)
(2007, 28, 28, 3)
fig,axs = plt.subplots(3,3,figsize=(14,14))
for i in range(9):
    r=i//3
    c=i%3
    ax=axs[r][c].imshow(X_images[i])
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X_images, y, test_size = 0.2, random_state=90)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

X_train = X_train/255
X_test = X_test/255

X_train = X_train.reshape(-1,28,28,3).astype('float32')
X_test = X_test.reshape(-1,28,28,3).astype('float32')
test_images2 = test_images/255
test = test_images2.reshape(-1,28,28,3).astype('float32')

generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                            rotation_range=10,
                                                            zoom_range=0.10,
                                                            width_shift_range=0.1,
                                                            height_shift_range=0.1,
                                                            shear_range=0.1,
                                                            horizontal_flip=False,
                                                            fill_mode="nearest")





model = Sequential()

model.add(Conv2D(128,(2,2),input_shape = (28,28,3),activation = 'relu'))
model.add(Conv2D(128,(2,2),activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(256,(2,2),activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(512,(2,2),activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(512,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

result = model.fit(X_train,y_train, validation_split=0.2, epochs=20, batch_size=256, verbose=2)
history_df = pd.DataFrame(result.history)
history_df.loc[:,['accuracy','val_accuracy']].plot()
history_df.loc[:,['loss','val_loss']].plot()
plt.show()




