import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image

np.random.seed(42)

# Parameters

# image width/height
IMG_WIDTH = 28
IMG_HEIGHT = 28
MODEL_INPUT_IMG_WIDTH = 32
MODEL_INPUT_IMG_HEIGHT = 32

# number of target classes
CLASS_NUM = 7

# learning batch size
BATCH_SIZE =256

# split rate
RATE_TRAIN_TEST_SPLIT = 0.2
RATE_TRAIN_VAL_SPLIT = 0.2

OBJECTIVE_FUNCTION = 'categorical_crossentropy'

rgb_data = pd.read_csv('./data/hmnist_28_28_RGB.csv')
meta = pd.read_csv('./data/HAM10000_metadata.csv')

label = rgb_data["label"]
rgb_data.drop('label', axis=1, inplace=True)

plt.figure(figsize=(20, 30))

cnt = 0
num_plot = 5

for l in np.sort(label.unique()):
    for i in label[label == l][:num_plot].index:
        plt.subplot(CLASS_NUM, num_plot, cnt + 1)
        plt.imshow(np.array(rgb_data.iloc[i]).reshape(IMG_WIDTH, IMG_HEIGHT, 3))
        plt.title(l)
        plt.axis("off")
        cnt += 1

plt.show()

from sklearn.model_selection import train_test_split

X_train_val, X_test, y_train_val, y_test = train_test_split(rgb_data, label, test_size=RATE_TRAIN_TEST_SPLIT, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=RATE_TRAIN_VAL_SPLIT, random_state=42)
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_smote, y_smote = sm.fit_resample(X_train, y_train)
plt.figure(figsize = (5,4))
plt.bar(y_smote.unique(), y_smote.value_counts())
def reshape_1d_to_img_array(X):
    return np.array(X).reshape(len(X), IMG_WIDTH, IMG_HEIGHT, 3)

X_smote_train = reshape_1d_to_img_array(X_smote)
X_val = reshape_1d_to_img_array(X_val)
X_test = reshape_1d_to_img_array(X_test)
img_resize = tf.image.resize(X_smote_train[0], [MODEL_INPUT_IMG_WIDTH, MODEL_INPUT_IMG_HEIGHT], method='nearest')
plt.imshow(X_smote_train[0])
plt.imshow(img_resize)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input , decode_predictions


def resize_image(img_array):
    return tf.image.resize(img_array, [MODEL_INPUT_IMG_WIDTH, MODEL_INPUT_IMG_HEIGHT], method='nearest')


def preprocess_data(x):
    x = resize_image(x)
    x = preprocess_input(x)

    return x


train_data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
train_data_generator.fit(X_smote_train)

X_val = preprocess_data(X_val)
X_test = preprocess_data(X_test)
from tensorflow.keras.utils import to_categorical

y_smote = to_categorical(y_smote)
y_val = to_categorical(y_val)

from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (Dense, Dropout, GlobalAveragePooling2D)


base_model = ResNet50(weights='imagenet', include_top=False,
                      input_shape=(MODEL_INPUT_IMG_WIDTH, MODEL_INPUT_IMG_HEIGHT, 3))

x = base_model.output
x = Dropout(0.3)(x)
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)

outputs = Dense(CLASS_NUM, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=outputs)


for layer in model.layers[:100]:
    layer.trainable = False
for layer in model.layers[100:]:
    layer.trainable = True
model.compile(loss=OBJECTIVE_FUNCTION,
              #optimizer=keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
              optimizer=keras.optimizers.Adam(lr=1e-5),
              #optimizer=keras.optimizers.RMSprop(lr=1e-5),
              metrics=['accuracy'])

#model.summary()

cb_earlystopping = EarlyStopping(monitor='val_loss', patience=5)
cb_model_check_point = ModelCheckpoint(filepath='./data/my_model/skin_cancer_mnist_resnet50.h5',
                                       monitor='val_loss',
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=False,
                                       mode='min',
                                       period=1)

cb_reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-8)

history = model.fit_generator(train_data_generator.flow(X_smote_train, y_smote, batch_size=BATCH_SIZE),
                              epochs=50,
                              steps_per_epoch=X_smote_train.shape[0]//BATCH_SIZE,
                              validation_data=(X_val, y_val),
                              shuffle=True,
                              callbacks=[cb_earlystopping, cb_model_check_point, cb_reduce_lr])

plt.figure(figsize=(5, 5))
plt.subplot(211)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

y_pred_prob = model.predict(X_test)
y_pred= np.argmax(y_pred_prob, axis=1)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_pred, y_test)
sns.heatmap(cm, square=True, cbar=True, annot=True, cmap='Blues')
plt.xlabel("Prediction", fontsize=13)
plt.ylabel("Answer", fontsize=13)
plt.savefig('./data/confusion_matrix.png')