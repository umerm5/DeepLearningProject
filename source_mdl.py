import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Convolution2D, Flatten, Dropout, MaxPool2D, BatchNormalization, GlobalAveragePooling2D
from keras.layers import Dense, Activation, Input
from keras.initializers import random_normal
from keras.optimizers import RMSprop, SGD, Adam
from keras.regularizers import l2
from keras.models import Sequential, Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, Callback, TensorBoard
#from matplotlib import pyplot
from sklearn.model_selection import train_test_split
import pandas as pd
import os
#import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob
image_list = []


def create_Model(shape):

    model = Sequential()
    model.add(Convolution2D(32, kernel_size=(3, 3), padding="same", activation="relu", input_shape=shape))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(Convolution2D(64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(Convolution2D(128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(Convolution2D(256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(Convolution2D(512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(Convolution2D(1024, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(Convolution2D(1048, kernel_size=(3, 3), padding="same", activation="relu"))


    model.add(Flatten())
    model.add(Dense(1,kernel_regularizer=l2(0.5), activation="sigmoid"))

    return model


####Resizing the overall data to 128x128 rgb images
# for filename in glob.glob('E:\DL\project\ISIC-2017_Training_Data/*.jpg'): #assuming jpg
#     im = Image.open(filename)
#     imResize = im.resize((128, 128)).convert('L') ####same as vgg dataset dimension
#     image_list.append(np.array(imResize))
#
# im.show()
# x = np.reshape(image_list, ((len(image_list), 128,128,1)))
# np.save('data_128by128_GS', x)

#####Pre-Processing Part to make the data balanced
x = np.load('data_128by128.npy')
y = np.loadtxt("Target.csv", delimiter=",")
num = np.count_nonzero(y)
data = x[0:2*num,:]
labels = y[0:2*num,]
count1 = 0
count2 = 0
l = 0
for i in range(len(y)):
    if(y[i,]== 0 and count1<=num-1):
        data[l,:] = x[i,:]
        labels[l,] = y[i,]
        count1  = count1 + 1
        l = l+1
    elif(y[i,]== 1 and count2<=num-1):
        data[l, :] = x[i, :]
        labels[l,] = y[i,]
        count2 = count2 + 1
        l = l+1

batch_size = 32
num_classes = 2
epochs = 101

x = data
y = labels
datagen = ImageDataGenerator(rotation_range=90)
datagen.fit(x)
X_batch, y_batch = datagen.flow(x, y, batch_size=len(x)).next()
a = np.array(X_batch)
y_train = np.append(y,y_batch)
print(y_train.shape)
x_train = np.append(x,a)
x_train = np.reshape(x_train,((len(x)+len(X_batch)),128,128,3))
print(x_train.shape)

datagen1 = ImageDataGenerator(horizontal_flip=True)
datagen1.fit(x)
X_batch1, y_batch1 = datagen1.flow(x, y, batch_size=len(x)).next()
a1 = np.array(X_batch1)
y_train = np.append(y_train,y_batch1)
print(y_train.shape)
x_train = np.append(x_train,a1)
x_train = np.reshape(x_train,(len(y_train),128,128,3))
print(x_train.shape)
x_train = x_train.astype('float32')
x_train /= 255
print(x_train.shape[0], 'train samples')

X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.10, random_state=42)
model = create_Model(X_train[0].shape)
print(model.summary())
adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
model.compile(loss="binary_crossentropy", optimizer=adam, metrics=['accuracy'])
earlystop  = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs, callbacks=[earlystop],
                    verbose=1,
                    validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)
pre = model.predict(X_test)
print(pre)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

pd.DataFrame(history.history).to_csv("history.csv")
model.save('ML_project_CNN.h5')
model.save_weights('ML_project_source_CNN_weights.h5')