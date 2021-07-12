import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pandas as pd
import datetime

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, BatchNormalization, Dropout, Dense, MaxPool2D
from keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

'''
0: nv - Melanocytic nevi
1: mel - Melanoma
2: bkl - Benign keratosis-like lesions
3: bcc - Basal cell carcinoma
4: akiec - Actinic keratoses and intraepithelial carcinoma / Bowen's disease
5: vasc - Vascular lesions
6: df - Dermatofibroma



raw_data = pd.read_csv('./data/hmnist_28_28_RGB.csv')
raw_data = raw_data.sample(frac = 1)
data = raw_data.iloc[:,:-1]
labels = raw_data.iloc[:,-1:]

type_cancer = ['akiec','df','bkl','mel','nv','vasc','bcc']
counts = list(labels.value_counts())
plt.figure(figsize = (8,6))
sns.barplot(x = type_cancer, y = counts)
plt.show()
'''
raw_data = pd.read_csv('./data/hmnist_28_28_RGB.csv')
raw_data = raw_data.sort_values('label')
raw_data = raw_data.reset_index()

index0 = raw_data[raw_data['label'] == 0].index.values
index1 = raw_data[raw_data['label'] == 1].index.values
index2 = raw_data[raw_data['label'] == 2].index.values
index3 = raw_data[raw_data['label'] == 3].index.values
index5 = raw_data[raw_data['label'] == 5].index.values
index6 = raw_data[raw_data['label'] == 6].index.values

df_index0 = raw_data.iloc[int(min(index0)):int(max(index0)+1)]
df_index1 = raw_data.iloc[int(min(index1)):int(max(index1)+1)]
df_index2 = raw_data.iloc[int(min(index2)):int(max(index2)+1)]
df_index3 = raw_data.iloc[int(min(index3)):int(max(index3)+1)]
df_index5 = raw_data.iloc[int(min(index5)):int(max(index5)+1)]
df_index6 = raw_data.iloc[int(min(index6)):int(max(index6)+1)]


df_index0 = df_index0.append([df_index0]*17, ignore_index = True)
df_index1 = df_index1.append([df_index1]*15, ignore_index = True)
df_index2 = df_index2.append([df_index2]*5, ignore_index = True)
df_index3 = df_index3.append([df_index3]*52, ignore_index = True)
df_index5 = df_index5.append([df_index5]*45, ignore_index = True)
df_index6 = df_index6.append([df_index6]*5, ignore_index = True)

frames = [raw_data, df_index0, df_index1, df_index2, df_index3, df_index5, df_index6]

final_data = pd.concat(frames)
final_data.drop('index', inplace = True, axis = 1)
final_data = final_data.sample(frac = 1)
data = final_data.iloc[:,:-1]
labels = final_data.iloc[:,-1:]
type_cancer = ['akiec','df','bkl','mel','nv','vasc','bcc']
counts = list(labels.value_counts())
plt.figure(figsize = (8,6))
sns.barplot(x = type_cancer, y = counts)
plt.show()

X = np.array(data)
Y = np.array(labels)

# reshaping the data

X = X.reshape(-1,28,28,3)

print("SHAPE OF X IS: ", X.shape)
print("SHAPE OF Y IS: ", Y.shape)




def visualisePlots(X, Y, rows, columns):
    class_dicts = {
        0: 'nv',
        1: 'mel',
        2: 'bkl',
        3: 'bcc',
        4: 'akiec',
        5: 'vasc',
        6: 'df',
    }

    data = []
    target = []

    for i in range(rows * columns):
        data.append(X[i])
        target.append(Y[i])

    width = 10
    height = 10
    fig = plt.figure(figsize=(10, 10))
    for i in range(columns * rows):
        temp_img = array_to_img(data[i])
        fig.add_subplot(rows, columns, i + 1)
        plt.imshow(temp_img)
        plt.xticks([])
        plt.yticks([])
        plt.title(str(class_dicts[target[i][0]]))
    plt.show()


# using the above function

visualisePlots(X, Y, 3, 3)

X = (X-np.mean(X))/np.std(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

train_datagen = ImageDataGenerator(rescale = 1./255,
                                  rotation_range = 10,
                                  width_shift_range = 0.2,
                                  height_shift_range = 0.2,
                                  shear_range = 0.2,
                                  horizontal_flip = True,
                                  vertical_flip = True,
                                  fill_mode = 'nearest')
train_datagen.fit(X_train)

test_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen.fit(X_test)

train_data = train_datagen.flow(X_train, Y_train, batch_size = 64)
test_data = test_datagen.flow(X_test, Y_test, batch_size = 64)


model = Sequential()
model.add(Conv2D(16, kernel_size = (3,3), input_shape = (28, 28, 3), activation = 'relu', padding = 'same'))
model.add(Conv2D(32, kernel_size = (3,3), activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2)))

model.add(Conv2D(32, kernel_size = (3,3), activation = 'relu', padding = 'same'))
model.add(Conv2D(64, kernel_size = (3,3), activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2), padding = 'same'))

model.add(Conv2D(64, kernel_size = (3,3), activation = 'relu'))
model.add(Conv2D(64, kernel_size = (3,3), activation = 'relu', padding = 'same'))
model.add(MaxPool2D(pool_size = (2,2), padding = 'same'))

model.add(Flatten())

model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(7, activation='softmax'))


learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)


optimizer = tf.keras.optimizers.Adam(learning_rate = 0.00075,
                                    beta_1 = 0.9,
                                    beta_2 = 0.999,
                                    epsilon = 1e-8)

model.compile(loss = 'sparse_categorical_crossentropy',
             optimizer = optimizer,
              metrics = ['accuracy'])

print(model.summary())


start_model = datetime.datetime.now()

history = model.fit(X_train,
                    Y_train,
                    validation_split=0.2,
                    batch_size = 512,
                    epochs = 30,
                    callbacks=[learning_rate_reduction])

end_model = datetime.datetime.now()


ACC = history.history['accuracy']
VAL_ACC = history.history['val_accuracy']

plt.figure(figsize=(8,6))
plt.title("THE ACCURACY OF THE TRAINING AND VALIDATION PHASE OF THE MODEL")
plt.plot(ACC, label = 'train_acc')
plt.plot(VAL_ACC, label = 'val_acc')
plt.legend()

LOSS = history.history['loss']
VAL_LOSS = history.history['val_loss']

plt.figure(figsize=(8,6))
plt.title("THE LOSS OF THE TRAINING AND VALIDATION PHASE OF THE MODEL")
plt.plot(LOSS, label = 'train_loss')
plt.plot(VAL_LOSS, label = 'val_loss')
plt.legend()


Y_true = np.array(Y_test)

Y_pred = model.predict(X_test)
Y_pred = np.array(list(map(lambda x: np.argmax(x), Y_pred)))

cm1 = confusion_matrix(Y_true, Y_pred)
plt.figure(figsize=(12, 6))
plt.title('####  THE CONFUSION MATRIX OF THE MODEL WITH TESTING DATA ####')
sns.heatmap(cm1, annot = True, fmt = 'g' ,vmin = 0, cmap = 'Blues')


def visualisePlots_test(X, Y, model, rows, columns):
    class_dicts = {
        0: 'nv',
        1: 'mel',
        2: 'bkl',
        3: 'bcc',
        4: 'akiec',
        5: 'vasc',
        6: 'df',
    }

    data = []
    target = []

    Y_pred = model.predict(X)
    Y_pred = np.array(list(map(lambda x: np.argmax(x), Y_pred)))

    for i in range(rows * columns):
        data.append(X[i])
        target.append(Y[i])

    width = 10
    height = 10
    fig = plt.figure(figsize=(10, 10))
    for i in range(columns * rows):
        temp_img = array_to_img(data[i])
        fig.add_subplot(rows, columns, i + 1)
        plt.imshow(temp_img)
        plt.xticks([])
        plt.yticks([])
        plt.title(str(class_dicts[target[i][0]]) + " || " + str(class_dicts[Y_pred[i]]))
    plt.show()

print('THE PLOTS TESTING WITH THE MODEL')
visualisePlots_test(X_test,Y_test, model, 3, 3)


label_mapping = {
    0: 'nv',
    1: 'mel',
    2: 'bkl',
    3: 'bcc',
    4: 'akiec',
    5: 'vasc',
    6: 'df'
}

classification_report_model = classification_report(Y_true, Y_pred, target_names=label_mapping.values())
print(classification_report_model)



time_model = end_model - start_model
print("TIME TAKEN BY MODEL : ", time_model)


model_acc_test = model.evaluate(X_test, Y_test, verbose=0)[1]
print("TEST ACCURACY OF MODEL: {:.3f}%".format(model_acc_test * 100))